package PrefixTabularRL.test

import PrefixAdderLib.DependentTopology
import PrefixRLCore.{DecisionContext, NormalizedObjectives, SearchPolicyState}
import PrefixTabularRL._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.Random

class TabularPolicySpec extends AnyFlatSpec with Matchers {
  private val existingOutputs = DependentTopology.balanced(8).outputs.take(5)
  private val width = 8
  private val context = DecisionContext(outputIndex = 4, segmentLow = 0, segmentHigh = 4, root = true)

  private def frontierState(
    completedEvaluations: Int,
    frontierSize:         Int,
    duplicateCount:       Int,
    frontierSpread:       Double
  ): SearchPolicyState = {
    val completed = math.max(1, completedEvaluations)
    SearchPolicyState(
      completedEvaluations = completedEvaluations,
      frontierSize = frontierSize,
      cacheSize = completed - duplicateCount,
      duplicateCount = duplicateCount,
      hasObservations = completedEvaluations > 0,
      hasFrontier = frontierSize > 0,
      duplicateRate = duplicateCount.toDouble / completed.toDouble,
      frontierFraction = frontierSize.toDouble / completed.toDouble,
      uniqueFraction = (completed - duplicateCount).toDouble / completed.toDouble,
      normalizedBest = NormalizedObjectives(0.1, 0.2, 0.3),
      normalizedMean = NormalizedObjectives(0.25, 0.35, 0.45),
      normalizedWorst = NormalizedObjectives(0.55, 0.7, 0.8),
      frontierSpread = frontierSpread,
      lastReward = 0.6,
      lastCacheHit = false
    )
  }

  private val coldState = SearchPolicyState.empty
  private val warmState =
    frontierState(completedEvaluations = 16, frontierSize = 4, duplicateCount = 5, frontierSpread = 0.4)

  "TabularSoftmaxPolicy" should "produce a valid probability distribution" in {
    val policy = new TabularSoftmaxPolicy(
      random = new Random(1),
      temperature = 1.0
    )

    val decision = policy.sample(warmState, width, context, existingOutputs)
    decision.probabilities.length shouldBe context.actionCount
    decision.probabilities.sum shouldBe 1.0 +- 1e-9
    decision.probabilities.foreach { p =>
      p should be >= 0.0
      p should be <= 1.0
    }
  }

  it should "increase the probability of rewarded actions after updates" in {
    val policy = new TabularSoftmaxPolicy(
      random = new Random(7),
      temperature = 1.0
    )

    val targetAction = 0
    val key = TabularContextKey(context, SearchStateBucket.from(warmState))
    val before = policy.distribution(context, warmState)(targetAction)

    for (_ <- 0 until 40) {
      val step = TabularDecision(
        tableKey = key,
        context = context,
        actionIndex = targetAction,
        probabilities = policy.distribution(context, warmState)
      )
      policy.update(
        trace = Seq(step),
        reward = 1.0,
        baseline = 0.0,
        learningRate = 0.05
      )
    }

    val after = policy.distribution(context, warmState)(targetAction)
    after should be > before
  }

  it should "maintain separate tables for different search buckets" in {
    val policy = new TabularSoftmaxPolicy(
      random = new Random(17),
      temperature = 1.0
    )

    val warmKey = TabularContextKey(context, SearchStateBucket.from(warmState))
    val coldBefore = policy.distribution(context, coldState)(0)

    for (_ <- 0 until 30) {
      val step = TabularDecision(
        tableKey = warmKey,
        context = context,
        actionIndex = 0,
        probabilities = policy.distribution(context, warmState)
      )
      policy.update(
        trace = Seq(step),
        reward = 1.0,
        baseline = 0.0,
        learningRate = 0.05
      )
    }

    val warmAfter = policy.distribution(context, warmState)(0)
    val coldAfter = policy.distribution(context, coldState)(0)

    warmAfter should be > coldAfter
    coldAfter shouldBe coldBefore +- 1e-9
  }

  it should "use an explicit saturation bucket for unit-interval search statistics" in {
    val state0 = frontierState(completedEvaluations = 8, frontierSize = 0, duplicateCount = 0, frontierSpread = 0.1)
    SearchStateBucket.from(state0).frontierBucket shouldBe 0
    SearchStateBucket.from(state0).duplicateBucket shouldBe 0

    val stateQuarter =
      frontierState(completedEvaluations = 8, frontierSize = 2, duplicateCount = 0, frontierSpread = 0.1)
    SearchStateBucket.from(stateQuarter).frontierBucket shouldBe 1

    val stateHalf = frontierState(completedEvaluations = 8, frontierSize = 4, duplicateCount = 0, frontierSpread = 0.1)
    SearchStateBucket.from(stateHalf).frontierBucket shouldBe 2

    val stateThreeQuarter =
      frontierState(completedEvaluations = 8, frontierSize = 6, duplicateCount = 0, frontierSpread = 0.1)
    SearchStateBucket.from(stateThreeQuarter).frontierBucket shouldBe 3

    val stateFullFrontier =
      frontierState(completedEvaluations = 8, frontierSize = 8, duplicateCount = 0, frontierSpread = 0.1)
    SearchStateBucket.from(stateFullFrontier).frontierBucket shouldBe 4

    val stateFullDuplicate =
      frontierState(completedEvaluations = 8, frontierSize = 0, duplicateCount = 8, frontierSpread = 0.1)
    SearchStateBucket.from(stateFullDuplicate).duplicateBucket shouldBe 4
  }

  it should "round-trip through JSON checkpoints" in {
    val policy = new TabularSoftmaxPolicy(
      random = new Random(23),
      temperature = 0.75
    )

    for (_ <- 0 until 12) {
      val step = policy.sample(warmState, width, context, existingOutputs)
      policy.update(
        trace = Seq(step.copy(actionIndex = 0)),
        reward = 1.0,
        baseline = 0.2,
        learningRate = 0.04
      )
    }

    val beforeWarm = policy.distribution(context, warmState)
    val beforeCold = policy.distribution(context, coldState)

    val dir = os.temp.dir(prefix = "tabular_policy_roundtrip_")
    val path = policy.write(dir / "policy.json")
    val loaded = TabularSoftmaxPolicy.load(path, new Random(99))

    loaded.temperatureValue shouldBe 0.75 +- 1e-9
    loaded.distribution(context, warmState) shouldBe beforeWarm
    loaded.distribution(context, coldState) shouldBe beforeCold
  }

  it should "write checkpoints as JSON" in {
    val policy = new TabularSoftmaxPolicy(
      random = new Random(3),
      temperature = 0.75
    )

    policy.sample(warmState, width, context, existingOutputs)

    val dir = os.temp.dir(prefix = "tabular_policy_spec_")
    val path = policy.write(dir / "policy.json")
    os.exists(path) shouldBe true

    val json = ujson.read(os.read(path))
    json("policy").str shouldBe "tabular-softmax-frontier-bucketed"
    json("temperature").num shouldBe 0.75 +- 1e-9
    json("context_count").num.toInt should be >= 1
    json("contexts").arr.head("search_bucket").obj.contains("phase_bucket") shouldBe true
  }
}
