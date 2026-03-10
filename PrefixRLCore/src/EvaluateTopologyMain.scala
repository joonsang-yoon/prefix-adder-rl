package PrefixRLCore

import PrefixAdderLib.DependentTopology
import mainargs.{arg, main, ParserForMethods}

object EvaluateTopologyMain {
  @main
  def run(
    @arg(name = "width", doc = "Adder width in bits")
    width: Int,
    @arg(name = "topology", doc = "Path to a dependent-tree topology JSON file")
    topology: String,
    @arg(name = "backend", doc = "Evaluation backend: librelane or synthetic")
    backend: String = sys.env.getOrElse("BACKEND", "librelane"),
    @arg(name = "output-root", doc = "Directory for generated RTL and LibreLane artifacts")
    outputRoot: String = "generated/eval",
    @arg(name = "repo-root", doc = "Repository root")
    repoRoot: String = ".",
    @arg(name = "librelane-config", doc = "Path to librelane_config.json (ignored by the synthetic backend)")
    librelaneConfig: String = "librelane_config.json",
    @arg(
      name = "elaborate-script",
      doc = "Path to scripts/elaborate_prefix_adder.sh (ignored by the synthetic backend)"
    )
    elaborateScript: String = "scripts/elaborate_prefix_adder.sh",
    @arg(name = "librelane-cmd", doc = "LibreLane command (ignored by the synthetic backend)")
    librelaneCmd: String = sys.env.getOrElse("LIBRELANE_CMD", "librelane"),
    @arg(name = "register-outputs", doc = "Wrap outputs in registers")
    registerOutputs: Boolean = true,
    @arg(name = "clock-period", doc = "Optional CLOCK_PERIOD override")
    clockPeriod: Double = 5.0
  ): Unit = {
    val repoRootPath = os.Path(repoRoot, os.pwd)
    val topologyPath = os.Path(topology, repoRootPath)
    val topo = DependentTopology.fromFile(topologyPath)
    require(topo.width == width, s"topology width ${topo.width} does not match --width ${width}")

    val evaluator = SearchSupport.createEvaluator(
      backend = backend,
      repoRoot = repoRootPath,
      workRoot = os.Path(outputRoot, repoRootPath),
      librelaneConfig = librelaneConfig,
      elaborateScript = elaborateScript,
      librelaneCmd = librelaneCmd,
      registerOutputs = registerOutputs,
      clockPeriod = clockPeriod
    )

    val result = evaluator.evaluate(0, topo)
    val json = ujson.Obj(
      "backend" -> backend.trim.toLowerCase,
      "ppa" -> result.ppa.toJson,
      "topology_path" -> result.topologyPath.toString,
      "rtl_dir" -> result.rtlDir.toString,
      "metrics_path" -> ujson.Str(result.metricsPath.map(_.toString).getOrElse(""))
    )
    Console.out.println(ujson.write(json, indent = 2))
  }

  def main(args: Array[String]): Unit = ParserForMethods(this).runOrExit(args.toIndexedSeq)
}
