package PrefixRLCore.test

import PrefixAdderLib.DependentTopology
import PrefixRLCore._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SearchSupportSpec extends AnyFlatSpec with Matchers {
  private final class StubEvaluator(ppa: PpaTuple) extends PhysicalEvaluator {
    var calls: Int = 0

    override def evaluate(episode: Int, topology: DependentTopology): PhysicalEvaluation = {
      calls += 1
      val root = os.temp.dir(prefix = "search_support_spec_eval_")
      PhysicalEvaluation(
        ppa = ppa,
        topologyPath = root / "topology.json",
        rtlDir = root / "rtl",
        metricsPath = Some(root / "metrics.json")
      )
    }
  }

  "SearchSupport.warmStarts" should "deduplicate identical deterministic seeds at small widths" in {
    val width2 = SearchSupport.warmStarts(2)
    width2 should have size 1
    width2.head.labels shouldBe Vector("ripple", "balanced")

    val width4 = SearchSupport.warmStarts(4)
    width4 should have size 2
    width4.flatMap(_.labels).toSet shouldBe Set("ripple", "balanced")
  }

  "SearchSupport.preparePaths" should "create a checkpoint directory and preserve existing files when cleanExisting is false" in {
    val root = os.temp.dir(prefix = "search_support_paths_")
    val first = SearchSupport.preparePaths(root, cleanExisting = true)
    os.write.over(first.logRoot / "sentinel.txt", "keep")

    val second = SearchSupport.preparePaths(root, cleanExisting = false)
    os.exists(second.checkpointRoot) shouldBe true
    os.read(second.logRoot / "sentinel.txt") shouldBe "keep"
  }

  it should "sanitize run labels when resolving output roots" in {
    val base = os.temp.dir(prefix = "search_support_label_")
    val resolved = SearchSupport.resolveOutputRoot(base, Some("baseline / sweep #1"))
    resolved.last shouldBe "baseline_sweep_1"
  }

  "SearchSupport.createEvaluator" should "support the synthetic backend without requiring LibreLane arguments" in {
    val repoRoot = os.temp.dir(prefix = "search_support_repo_root_")
    val workRoot = os.temp.dir(prefix = "search_support_work_root_")

    val evaluator = SearchSupport.createEvaluator(
      backend = "synthetic",
      repoRoot = repoRoot,
      workRoot = workRoot,
      librelaneConfig = "librelane_config.json",
      elaborateScript = "scripts/elaborate_prefix_adder.sh",
      librelaneCmd = "python -m librelane",
      registerOutputs = true,
      clockPeriod = 5.0
    )

    evaluator shouldBe a[SyntheticEvaluator]
  }

  "SearchEnvironment" should "reuse cached evaluations while assigning the current episode id" in {
    val evaluator = new StubEvaluator(PpaTuple(10.0, 5.0, 100.0))
    val logRoot = os.temp.dir(prefix = "search_environment_spec_")
    val env = new SearchEnvironment(evaluator, logRoot)
    val topology = DependentTopology.ripple(4)

    val first = env.evaluate(0, topology)
    val second = env.evaluate(5, topology)

    evaluator.calls shouldBe 1
    first.cacheHit shouldBe false
    second.cacheHit shouldBe true
    second.evaluated.episode shouldBe 5
    second.evaluated.id shouldBe s"5_${topology.stats.fingerprint}"
    second.evaluated.id should not be first.evaluated.id
    second.evaluated.topologyPath shouldBe first.evaluated.topologyPath
    second.evaluated.duplicate shouldBe true

    first.stateBefore.completedEvaluations shouldBe 0
    first.stateAfter.completedEvaluations shouldBe 1
    first.stateAfter.frontierSize shouldBe 1
    first.stateAfter.cacheSize shouldBe 1
    first.stateAfter.lastCacheHit shouldBe false

    second.stateBefore.completedEvaluations shouldBe 1
    second.stateAfter.completedEvaluations shouldBe 2
    second.stateAfter.frontierSize shouldBe 1
    second.stateAfter.cacheSize shouldBe 1
    second.stateAfter.duplicateCount shouldBe 1
    second.stateAfter.lastCacheHit shouldBe true

    env.policyState.completedEvaluations shouldBe 2
    env.policyState.duplicateCount shouldBe 1
  }
}
