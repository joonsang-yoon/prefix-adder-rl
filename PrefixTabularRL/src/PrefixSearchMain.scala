package PrefixTabularRL

import PrefixRLCore.{
  DependentTreeEpisodeBuilder,
  PolicyGradientSearchRunner,
  SearchEnvironment,
  SearchRunConfig,
  SearchSupport
}
import PrefixUtils.Catalan
import mainargs.{arg, main, ParserForMethods}

import scala.util.Random

object PrefixSearchMain {
  @main
  def run(
    @arg(name = "width", doc = "Adder width in bits")
    width: Int = 8,
    @arg(name = "episodes", doc = "Number of policy-gradient episodes after deterministic warm-starts")
    episodes: Int = 32,
    @arg(name = "seed", doc = "Random seed")
    seed: Int = 1,
    @arg(name = "learning-rate", doc = "Policy-gradient learning rate")
    learningRate: Double = 0.08,
    @arg(name = "temperature", doc = "Softmax temperature for action sampling")
    temperature: Double = 1.0,
    @arg(name = "baseline-momentum", doc = "EMA momentum used for the scalar reward baseline")
    baselineMomentum: Double = 0.9,
    @arg(
      name = "checkpoint-interval",
      doc = "Write an additional immutable checkpoint every N training episodes; 0 disables periodic snapshots"
    )
    checkpointInterval: Int = 0,
    @arg(name = "policy-init", doc = "Optional path to a saved policy JSON checkpoint used for warm-starting")
    policyInit: String = "",
    @arg(name = "run-label", doc = "Optional subdirectory label appended under output-root")
    runLabel: String = "",
    @arg(
      name = "clean-output",
      doc = "Delete prior work/log subdirectories under the resolved output root before starting"
    )
    cleanOutput: Boolean = true,
    @arg(name = "skip-warm-starts", doc = "Disable deterministic ripple/balanced warm starts")
    skipWarmStarts: Boolean = false,
    @arg(name = "backend", doc = "Evaluation backend: librelane or synthetic")
    backend: String = sys.env.getOrElse("BACKEND", "librelane"),
    @arg(name = "output-root", doc = "Directory for generated RTL, metrics and episode logs")
    outputRoot: String = "generated/search/tabular",
    @arg(name = "repo-root", doc = "Repository root used when invoking helper scripts")
    repoRoot: String = ".",
    @arg(
      name = "librelane-config",
      doc = "Path to the shipped librelane_config.json template (ignored by the synthetic backend)"
    )
    librelaneConfig: String = "librelane_config.json",
    @arg(
      name = "elaborate-script",
      doc = "Path to scripts/elaborate_prefix_adder.sh (ignored by the synthetic backend)"
    )
    elaborateScript: String = "scripts/elaborate_prefix_adder.sh",
    @arg(
      name = "librelane-cmd",
      doc = "LibreLane command; for example 'librelane' or 'python -m librelane' (ignored by the synthetic backend)"
    )
    librelaneCmd: String = sys.env.getOrElse("LIBRELANE_CMD", "librelane"),
    @arg(name = "register-outputs", doc = "Wrap the combinational adder with registered outputs")
    registerOutputs: Boolean = true,
    @arg(name = "clock-period", doc = "Optional CLOCK_PERIOD override written into the copied LibreLane config")
    clockPeriod: Double = 5.0
  ): Unit = {
    require(width >= 1, s"width must be >= 1, got ${width}")
    require(episodes >= 0, s"episodes must be >= 0, got ${episodes}")
    require(checkpointInterval >= 0, s"checkpointInterval must be >= 0, got ${checkpointInterval}")

    val repoRootPath = os.Path(repoRoot, os.pwd)
    val resolvedOutputRoot = SearchSupport.resolveOutputRoot(
      os.Path(outputRoot, repoRootPath),
      Option(runLabel).map(_.trim).filter(_.nonEmpty)
    )
    val paths = SearchSupport.preparePaths(resolvedOutputRoot, cleanExisting = cleanOutput)

    val evaluator = SearchSupport.createEvaluator(
      backend = backend,
      repoRoot = repoRootPath,
      workRoot = paths.workRoot,
      librelaneConfig = librelaneConfig,
      elaborateScript = elaborateScript,
      librelaneCmd = librelaneCmd,
      registerOutputs = registerOutputs,
      clockPeriod = clockPeriod
    )

    val env = new SearchEnvironment(evaluator, paths.logRoot)
    val random = new Random(seed)
    val policy =
      Option(policyInit).map(_.trim).filter(_.nonEmpty) match {
        case Some(path) => TabularSoftmaxPolicy.load(os.Path(path, repoRootPath), random)
        case None       => new TabularSoftmaxPolicy(random, temperature)
      }
    val builder = new DependentTreeEpisodeBuilder(policy)
    val warmStarts = if (skipWarmStarts) Vector.empty else SearchSupport.warmStarts(width)

    SearchSupport.writeRunConfig(
      paths.logRoot,
      ujson.Obj(
        "algorithm" -> "tabular",
        "width" -> width,
        "episodes" -> episodes,
        "seed" -> seed,
        "learning_rate" -> learningRate,
        "temperature" -> policy.temperatureValue,
        "baseline_momentum" -> baselineMomentum,
        "checkpoint_interval" -> checkpointInterval,
        "policy_init" -> policyInit.trim,
        "run_label" -> runLabel.trim,
        "clean_output" -> cleanOutput,
        "skip_warm_starts" -> skipWarmStarts,
        "backend" -> backend.trim.toLowerCase,
        "output_root" -> resolvedOutputRoot.toString,
        "register_outputs" -> registerOutputs,
        "clock_period" -> clockPeriod
      )
    )

    println(s"[search] backend=${backend.trim.toLowerCase}")
    println(s"[search] output_root=${resolvedOutputRoot}")
    println(
      s"[search] dependent-tree search space extension count for width=${width}: ${Catalan.extensionCount(width)}"
    )
    println(s"[search] dependent-tree network count up to width=${width}: ${Catalan.dependentNetworkCount(width)}")
    println(
      s"[search] unique deterministic warm-starts for width=${width}: " +
        (if (warmStarts.isEmpty) "<disabled>" else warmStarts.map(_.label).mkString(", "))
    )

    PolicyGradientSearchRunner.run[TabularDecision, TabularTrainingStats](
      config = SearchRunConfig(
        algorithm = "tabular",
        width = width,
        episodes = episodes,
        logRoot = paths.logRoot,
        policyRoot = paths.policyRoot,
        checkpointRoot = paths.checkpointRoot,
        baselineMomentum = baselineMomentum,
        checkpointInterval = checkpointInterval
      ),
      env = env,
      warmStarts = warmStarts,
      sampleEpisode = searchState => builder.sample(width, searchState),
      updatePolicy = (trace, reward, baseline) => policy.update(trace, reward, baseline, learningRate),
      writePolicy = path => policy.write(path),
      trainingSummary = stats => f"advantage=${stats.advantage}%.4f entropy=${stats.averageEntropy}%.4f",
      trainingToJson = _.toJson,
      summaryFields = Seq(
        "seed" -> ujson.Num(seed),
        "backend" -> ujson.Str(backend.trim.toLowerCase),
        "learning_rate" -> ujson.Num(learningRate),
        "temperature" -> ujson.Num(policy.temperatureValue),
        "baseline_momentum" -> ujson.Num(baselineMomentum),
        "checkpoint_interval" -> ujson.Num(checkpointInterval),
        "policy_init" -> ujson.Str(policyInit.trim),
        "run_label" -> ujson.Str(runLabel.trim),
        "clean_output" -> ujson.Bool(cleanOutput),
        "skip_warm_starts" -> ujson.Bool(skipWarmStarts),
        "policy_architecture" -> ujson.Str("tabular-softmax-frontier-bucketed"),
        "register_outputs" -> ujson.Bool(registerOutputs),
        "clock_period" -> ujson.Num(clockPeriod),
        "extension_count" -> ujson.Str(Catalan.extensionCount(width).toString),
        "dependent_network_count" -> ujson.Str(Catalan.dependentNetworkCount(width).toString),
        "resolved_output_root" -> ujson.Str(resolvedOutputRoot.toString)
      )
    )
  }

  def main(args: Array[String]): Unit = ParserForMethods(this).runOrExit(args.toIndexedSeq)
}
