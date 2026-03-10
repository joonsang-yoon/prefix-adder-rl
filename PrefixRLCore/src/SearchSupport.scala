package PrefixRLCore

import PrefixAdderLib.DependentTopology
import PrefixUtils.ShellWords

import scala.collection.mutable

final case class SearchPaths(
  outputRoot:     os.Path,
  workRoot:       os.Path,
  logRoot:        os.Path,
  policyRoot:     os.Path,
  checkpointRoot: os.Path
)

final case class WarmStartDesign(
  labels:   Vector[String],
  topology: DependentTopology
) {
  def label: String = labels.mkString("+")
}

final case class SearchRunConfig(
  algorithm:          String,
  width:              Int,
  episodes:           Int,
  logRoot:            os.Path,
  policyRoot:         os.Path,
  checkpointRoot:     os.Path,
  baselineMomentum:   Double,
  checkpointInterval: Int
)

object SearchSupport {
  val SupportedBackends: Vector[String] = Vector("librelane", "synthetic")

  def preparePaths(outputRoot: os.Path, cleanExisting: Boolean = true): SearchPaths = {
    val workRoot = outputRoot / "designs"
    val logRoot = outputRoot / "logs"
    val policyRoot = logRoot / "policy"
    val checkpointRoot = policyRoot / "checkpoints"

    os.makeDir.all(outputRoot)

    if (cleanExisting) {
      if (os.exists(workRoot)) {
        os.remove.all(workRoot)
      }
      if (os.exists(logRoot)) {
        os.remove.all(logRoot)
      }
    }

    os.makeDir.all(workRoot)
    os.makeDir.all(checkpointRoot)

    SearchPaths(
      outputRoot = outputRoot,
      workRoot = workRoot,
      logRoot = logRoot,
      policyRoot = policyRoot,
      checkpointRoot = checkpointRoot
    )
  }

  def resolveOutputRoot(baseRoot: os.Path, runLabel: Option[String]): os.Path = {
    runLabel.map(sanitizeRunLabel).filter(_.nonEmpty) match {
      case Some(label) => baseRoot / label
      case None        => baseRoot
    }
  }

  def sanitizeRunLabel(label: String): String = {
    val normalized = label.trim
    if (normalized.isEmpty) {
      ""
    } else {
      normalized.map { ch =>
        if (ch.isLetterOrDigit || ch == '-' || ch == '_' || ch == '.') ch else '_'
      }.mkString
        .replaceAll("_+", "_")
        .stripPrefix("_")
        .stripSuffix("_")
    }
  }

  def appendJsonLine(path: os.Path, value: ujson.Value): Unit = {
    val line = ujson.write(value) + "\n"
    if (os.exists(path)) {
      os.write.append(path, line)
    } else {
      os.write.over(path, line)
    }
  }

  def writeRunConfig(logRoot: os.Path, config: ujson.Value): os.Path = {
    val path = logRoot / "run_config.json"
    os.write.over(path, ujson.write(config, indent = 2))
    path
  }

  def createEvaluator(
    backend:         String,
    repoRoot:        os.Path,
    workRoot:        os.Path,
    librelaneConfig: String,
    elaborateScript: String,
    librelaneCmd:    String,
    registerOutputs: Boolean,
    clockPeriod:     Double
  ): PhysicalEvaluator = {
    backend.trim.toLowerCase match {
      case "librelane" =>
        new LibreLaneEvaluator(
          LibreLaneBackendConfig(
            repoRoot = repoRoot,
            workRoot = workRoot,
            configTemplate = os.Path(librelaneConfig, repoRoot),
            elaborateScript = os.Path(elaborateScript, repoRoot),
            command = ShellWords.split(librelaneCmd),
            registerOutputs = registerOutputs,
            clockPeriodOverride = Some(clockPeriod)
          )
        )
      case "synthetic" =>
        new SyntheticEvaluator(
          SyntheticBackendConfig(
            workRoot = workRoot,
            registerOutputs = registerOutputs,
            clockPeriod = clockPeriod
          )
        )
      case other =>
        throw new IllegalArgumentException(
          s"Unsupported backend '${other}'. Expected one of: ${SupportedBackends.mkString(", ")}"
        )
    }
  }

  def warmStarts(width: Int): Vector[WarmStartDesign] = {
    val byFingerprint = mutable.LinkedHashMap.empty[String, WarmStartDesign]
    val candidates = Vector(
      "ripple" -> DependentTopology.ripple(width),
      "balanced" -> DependentTopology.balanced(width)
    )

    candidates.foreach { case (label, topology) =>
      val fingerprint = topology.stats.fingerprint
      byFingerprint.get(fingerprint) match {
        case Some(existing) =>
          byFingerprint.update(fingerprint, existing.copy(labels = existing.labels :+ label))
        case None =>
          byFingerprint.update(fingerprint, WarmStartDesign(labels = Vector(label), topology = topology))
      }
    }

    byFingerprint.values.toVector
  }

  def updateBaseline(previous: Double, reward: Double, completedEpisodes: Int, momentum: Double): Double = {
    require(momentum >= 0.0 && momentum < 1.0, s"baseline momentum must be in [0, 1), got ${momentum}")

    if (completedEpisodes == 0) {
      reward
    } else {
      momentum * previous + (1.0 - momentum) * reward
    }
  }
}

object PolicyGradientSearchRunner {
  def run[D <: SplitDecision, T](
    config:          SearchRunConfig,
    env:             SearchEnvironment,
    warmStarts:      Vector[WarmStartDesign],
    sampleEpisode:   SearchPolicyState => EpisodeSample[D],
    updatePolicy:    (Seq[D], Double, Double) => T,
    writePolicy:     os.Path => os.Path,
    trainingSummary: T => String,
    trainingToJson:  T => ujson.Value,
    summaryFields:   Seq[(String, ujson.Value)]
  ): Unit = {
    require(config.width >= 1, s"width must be >= 1, got ${config.width}")
    require(config.episodes >= 0, s"episodes must be >= 0, got ${config.episodes}")
    require(
      config.baselineMomentum >= 0.0 && config.baselineMomentum < 1.0,
      s"baseline momentum must be in [0, 1), got ${config.baselineMomentum}"
    )
    require(config.checkpointInterval >= 0, s"checkpointInterval must be >= 0, got ${config.checkpointInterval}")

    val trainingLogPath = config.logRoot / "training.jsonl"

    val warmStartRewards = warmStarts.zipWithIndex.map { case (warmStart, episode) =>
      val result = env.evaluate(episode, warmStart.topology)
      SearchSupport.appendJsonLine(
        trainingLogPath,
        ujson.Obj(
          "phase" -> "warm-start",
          "episode_index" -> episode,
          "warm_start_label" -> warmStart.label,
          "reward" -> result.reward.reward,
          "power" -> result.evaluated.ppa.power,
          "delay" -> result.evaluated.ppa.delay,
          "area" -> result.evaluated.ppa.area,
          "frontier_size" -> result.update.frontierSize,
          "added_to_frontier" -> result.update.addedToFrontier,
          "cache_hit" -> result.cacheHit,
          "duplicate" -> result.update.duplicate
        )
      )
      println(
        f"[warm-start:${warmStart.label}] reward=${result.reward.reward}%.4f " +
          f"power=${result.evaluated.ppa.power}%.4f delay=${result.evaluated.ppa.delay}%.4f area=${result.evaluated.ppa.area}%.4f " +
          s"frontier_added=${result.update.addedToFrontier}"
      )
      result.reward.reward
    }

    val initialBaseline = if (warmStartRewards.nonEmpty) {
      warmStartRewards.sum / warmStartRewards.length.toDouble
    } else {
      0.0
    }
    var baseline = initialBaseline

    var completed = 0
    while (completed < config.episodes) {
      val episodeIndex = warmStarts.size + completed
      val baselineBefore = baseline
      val sample = sampleEpisode(env.policyState)
      val result = env.evaluate(episodeIndex, sample.topology)
      val training = updatePolicy(sample.decisions, result.reward.reward, baselineBefore)
      baseline = SearchSupport.updateBaseline(
        baselineBefore,
        result.reward.reward,
        warmStarts.length + completed,
        config.baselineMomentum
      )
      completed += 1

      val latestPolicyPath = writePolicy(config.policyRoot / "policy_latest.json")
      val checkpointPath =
        if (config.checkpointInterval > 0 && completed % config.checkpointInterval == 0)
          Some(writePolicy(config.checkpointRoot / f"policy_ep${episodeIndex}%05d.json"))
        else None

      SearchSupport.appendJsonLine(
        trainingLogPath,
        ujson.Obj(
          "phase" -> "train",
          "episode_index" -> episodeIndex,
          "completed_train_episodes" -> completed,
          "reward" -> result.reward.reward,
          "baseline_before" -> baselineBefore,
          "baseline_after" -> baseline,
          "power" -> result.evaluated.ppa.power,
          "delay" -> result.evaluated.ppa.delay,
          "area" -> result.evaluated.ppa.area,
          "frontier_size" -> result.update.frontierSize,
          "added_to_frontier" -> result.update.addedToFrontier,
          "cache_hit" -> result.cacheHit,
          "duplicate" -> result.update.duplicate,
          "latest_policy_checkpoint" -> latestPolicyPath.toString,
          "periodic_checkpoint" -> checkpointPath.fold[ujson.Value](ujson.Null)(p => ujson.Str(p.toString)),
          "training" -> trainingToJson(training)
        )
      )

      println(
        f"[episode ${episodeIndex}%05d] reward=${result.reward.reward}%.4f baseline=${baseline}%.4f " +
          s"${trainingSummary(training)} " +
          f"power=${result.evaluated.ppa.power}%.4f delay=${result.evaluated.ppa.delay}%.4f area=${result.evaluated.ppa.area}%.4f " +
          s"frontier_size=${result.update.frontierSize} cache_hit=${result.cacheHit}"
      )
    }

    val finalPolicyPath = writePolicy(config.policyRoot / "policy_final.json")

    val summary = ujson.Obj(
      "algorithm" -> config.algorithm,
      "width" -> config.width,
      "episodes" -> config.episodes,
      "warm_start_count" -> warmStarts.length,
      "total_evaluations" -> (warmStarts.length + config.episodes),
      "initial_baseline" -> initialBaseline,
      "final_baseline" -> baseline,
      "baseline_momentum" -> config.baselineMomentum,
      "checkpoint_interval" -> config.checkpointInterval,
      "training_log" -> trainingLogPath.toString,
      "policy_checkpoint" -> finalPolicyPath.toString,
      "warm_starts" -> ujson.Arr.from(warmStarts.map { warmStart =>
        ujson.Obj(
          "label" -> warmStart.label,
          "labels" -> ujson.Arr.from(warmStart.labels.map(label => ujson.Str(label))),
          "fingerprint" -> warmStart.topology.stats.fingerprint,
          "topology" -> warmStart.topology.toJson
        )
      }),
      "final_search_state" -> env.policyState.toJson,
      "frontier" -> ujson.Arr.from(env.frontierSnapshot.map(_.toJson))
    )

    summaryFields.foreach { case (key, value) =>
      summary.obj.update(key, value)
    }

    os.write.over(config.logRoot / "summary.json", ujson.write(summary, indent = 2))
  }
}
