package PrefixRLCore

import PrefixAdderLib.DependentTopology
import PrefixUtils.JsonSupport

final case class LibreLaneBackendConfig(
  repoRoot:            os.Path,
  workRoot:            os.Path,
  configTemplate:      os.Path,
  elaborateScript:     os.Path,
  command:             Seq[String],
  registerOutputs:     Boolean,
  clockPeriodOverride: Option[Double]
)

final case class PhysicalEvaluation(
  ppa:          PpaTuple,
  topologyPath: os.Path,
  rtlDir:       os.Path,
  metricsPath:  Option[os.Path]
)

trait PhysicalEvaluator {
  def evaluate(episode: Int, topology: DependentTopology): PhysicalEvaluation
}

final class LibreLaneEvaluator(cfg: LibreLaneBackendConfig) extends PhysicalEvaluator {
  override def evaluate(episode: Int, topology: DependentTopology): PhysicalEvaluation = {
    os.makeDir.all(cfg.workRoot)

    val runName = f"ep${episode}%05d_${topology.stats.fingerprint}"
    val designDir = cfg.workRoot / runName
    val rtlDir = designDir / "rtl"
    val topologyPath = designDir / "topology.json"
    val configPath = designDir / "librelane_config.json"
    val dotPath = designDir / "topology.dot"

    if (os.exists(designDir)) {
      os.remove.all(designDir)
    }
    os.makeDir.all(rtlDir)

    topology.write(topologyPath)
    os.write.over(dotPath, topology.toDot)
    writeDesignConfig(configPath)

    elaborate(topology, topologyPath, rtlDir)
    runLibreLane(configPath)

    val metricsPath = findMetricsJson(designDir)
    val ppa = metricsPath match {
      case Some(path) => parsePpa(path)
      case None =>
        throw new IllegalStateException(s"LibreLane completed without producing a metrics.json under ${designDir}")
    }

    PhysicalEvaluation(
      ppa = ppa,
      topologyPath = topologyPath,
      rtlDir = rtlDir,
      metricsPath = metricsPath
    )
  }

  private def writeDesignConfig(path: os.Path): Unit = {
    val base = ujson.read(os.read(cfg.configTemplate))
    val patched = cfg.clockPeriodOverride match {
      case Some(clockPeriod) =>
        base.obj.update("CLOCK_PERIOD", ujson.Num(clockPeriod))
        base
      case None => base
    }
    os.write.over(path, ujson.write(patched, indent = 2))
  }

  private def elaborate(
    topology:     DependentTopology,
    topologyPath: os.Path,
    rtlDir:       os.Path
  ): Unit = {
    val cmd = Seq(
      "bash",
      cfg.elaborateScript.toString,
      "--width",
      topology.width.toString,
      "--topology",
      topologyPath.toString,
      "--register-outputs",
      cfg.registerOutputs.toString,
      "--target-dir",
      rtlDir.toString
    )
    os.proc(cmd).call(cwd = cfg.repoRoot)
  }

  private def runLibreLane(configPath: os.Path): Unit = {
    if (cfg.command.isEmpty) {
      throw new IllegalArgumentException("LibreLane command must not be empty")
    }
    os.proc(cfg.command ++ Seq(configPath.toString)).call(cwd = cfg.repoRoot)
  }

  private def findMetricsJson(designDir: os.Path): Option[os.Path] = {
    val preferred = os.walk(designDir).filter(_.last == "metrics.json")
    val fallback = os.walk(designDir).filter(_.last == "or_metrics_out.json")
    val all = if (preferred.nonEmpty) preferred else fallback
    all.sortBy(path => -os.mtime(path)).headOption
  }

  private def parsePpa(metricsPath: os.Path): PpaTuple = {
    val raw = ujson.read(os.read(metricsPath))
    val metrics = raw.obj.get("metrics").getOrElse(raw)

    def metric(name: String): Double = {
      metrics.obj.get(name) match {
        case Some(value) => JsonSupport.readDouble(value)
        case None        => throw new NoSuchElementException(s"Missing metric '${name}' in ${metricsPath}")
      }
    }

    val power = metric("power__total")
    val delay = metric("clock_period") - metric("timing__setup__ws")
    val area = metric("design__instance__area")
    PpaTuple(power = power, delay = delay, area = area)
  }
}
