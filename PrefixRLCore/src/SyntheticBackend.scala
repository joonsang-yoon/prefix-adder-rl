package PrefixRLCore

import PrefixAdderLib.DependentTopology

final case class SyntheticBackendConfig(
  workRoot:        os.Path,
  registerOutputs: Boolean,
  clockPeriod:     Double
)

final class SyntheticEvaluator(cfg: SyntheticBackendConfig) extends PhysicalEvaluator {
  require(cfg.clockPeriod.isFinite && cfg.clockPeriod > 0.0, s"clockPeriod must be > 0, got ${cfg.clockPeriod}")

  override def evaluate(episode: Int, topology: DependentTopology): PhysicalEvaluation = {
    os.makeDir.all(cfg.workRoot)

    val runName = f"ep${episode}%05d_${topology.stats.fingerprint}"
    val designDir = cfg.workRoot / runName
    val rtlDir = designDir / "rtl"
    val topologyPath = designDir / "topology.json"
    val metricsPath = designDir / "metrics.json"
    val dotPath = designDir / "topology.dot"

    if (os.exists(designDir)) {
      os.remove.all(designDir)
    }
    os.makeDir.all(rtlDir)

    topology.write(topologyPath)
    os.write.over(dotPath, topology.toDot)

    val ppa = SyntheticSurrogateModel.estimate(topology, cfg.registerOutputs)
    val metrics = SyntheticSurrogateModel.toMetricsJson(topology, ppa, cfg.clockPeriod, cfg.registerOutputs)
    os.write.over(metricsPath, ujson.write(metrics, indent = 2))
    os.write.over(rtlDir / "README.txt", placeholderRtlNote(topology))

    PhysicalEvaluation(
      ppa = ppa,
      topologyPath = topologyPath,
      rtlDir = rtlDir,
      metricsPath = Some(metricsPath)
    )
  }

  private def placeholderRtlNote(topology: DependentTopology): String = {
    val stats = topology.stats
    s"""|Synthetic backend placeholder
        |
        |This run intentionally skipped Chisel elaboration and LibreLane.
        |The persisted metrics.json file contains deterministic surrogate PPA values
        |derived from topology statistics only.
        |
        |Topology fingerprint: ${stats.fingerprint}
        |Width: ${topology.width}
        |Unique internal nodes: ${stats.uniqueInternalNodes}
        |Total internal nodes: ${stats.totalInternalNodes}
        |Max depth: ${stats.maxDepth}
        |Average depth: ${stats.averageDepth}
        |Reuse ratio: ${stats.reuseRatio}
        |Register outputs: ${cfg.registerOutputs}
        |Clock period: ${cfg.clockPeriod}
        |""".stripMargin
  }
}

private object SyntheticSurrogateModel {
  def estimate(topology: DependentTopology, registerOutputs: Boolean): PpaTuple = {
    val stats = topology.stats
    val width = topology.width.toDouble
    val unique = stats.uniqueInternalNodes.toDouble
    val total = stats.totalInternalNodes.toDouble
    val maxDepth = stats.maxDepth.toDouble
    val averageDepth = stats.averageDepth
    val reuseRatio = stats.reuseRatio

    val registerPower = if (registerOutputs) 0.18 * width else 0.0
    val registerDelay = if (registerOutputs) -0.35 else 0.0
    val registerArea = if (registerOutputs) 1.75 * width else 0.0

    val power =
      0.5 * width +
        0.7 * unique +
        0.15 * total +
        0.25 * averageDepth +
        0.3 * (1.0 - reuseRatio) +
        registerPower +
        0.05 * fingerprintNoise(stats.fingerprint, 0)

    val delay =
      0.35 * width +
        0.95 * maxDepth +
        0.35 * averageDepth +
        0.1 * (unique / math.max(1.0, width)) +
        registerDelay +
        0.05 * fingerprintNoise(stats.fingerprint, 8)

    val area =
      10.0 * width +
        3.8 * unique +
        0.65 * total +
        0.4 * maxDepth +
        registerArea +
        0.15 * fingerprintNoise(stats.fingerprint, 4)

    PpaTuple(
      power = power,
      delay = math.max(0.05, delay),
      area = area
    )
  }

  def toMetricsJson(
    topology:        DependentTopology,
    ppa:             PpaTuple,
    clockPeriod:     Double,
    registerOutputs: Boolean
  ): ujson.Value = {
    val stats = topology.stats
    val slack = clockPeriod - ppa.delay

    ujson.Obj(
      "backend" -> "synthetic",
      "mode" -> "surrogate",
      "register_outputs" -> registerOutputs,
      "topology_fingerprint" -> stats.fingerprint,
      "metrics" -> ujson.Obj(
        "power__total" -> ppa.power,
        "clock_period" -> clockPeriod,
        "timing__setup__ws" -> slack,
        "design__instance__area" -> ppa.area
      ),
      "surrogate_components" -> ujson.Obj(
        "width" -> topology.width,
        "unique_internal_nodes" -> stats.uniqueInternalNodes,
        "total_internal_nodes" -> stats.totalInternalNodes,
        "max_depth" -> stats.maxDepth,
        "average_depth" -> stats.averageDepth,
        "reuse_ratio" -> stats.reuseRatio,
        "power_noise" -> fingerprintNoise(stats.fingerprint, 0),
        "delay_noise" -> fingerprintNoise(stats.fingerprint, 8),
        "area_noise" -> fingerprintNoise(stats.fingerprint, 4)
      )
    )
  }

  private def fingerprintNoise(fingerprint: String, start: Int): Double = {
    if (fingerprint.isEmpty) {
      0.0
    } else {
      val normalizedStart = Math.floorMod(start, fingerprint.length)
      val rotated = fingerprint.drop(normalizedStart) + fingerprint.take(normalizedStart)
      val slice = rotated.take(math.min(8, rotated.length))
      BigInt(slice, 16).toDouble / 0xffffffffL.toDouble
    }
  }
}
