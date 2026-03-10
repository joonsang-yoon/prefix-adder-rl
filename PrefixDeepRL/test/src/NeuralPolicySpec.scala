package PrefixDeepRL.test

import PrefixAdderLib.DependentTopology
import PrefixDeepRL._
import PrefixRLCore.{DecisionContext, NormalizedObjectives, SearchPolicyState}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.Random

class NeuralPolicySpec extends AnyFlatSpec with Matchers {
  private val width = 8
  private val topology = DependentTopology.balanced(width)
  private val context = DecisionContext(outputIndex = 4, segmentLow = 0, segmentHigh = 4, root = true)
  private val existingOutputs = topology.outputs.take(context.outputIndex)

  private def frontierState(
    completedEvaluations: Int,
    frontierSize:         Int,
    duplicateCount:       Int,
    frontierSpread:       Double,
    lastReward:           Double,
    lastCacheHit:         Boolean
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
      normalizedBest = NormalizedObjectives(0.05, 0.15, 0.25),
      normalizedMean = NormalizedObjectives(0.2, 0.35, 0.45),
      normalizedWorst = NormalizedObjectives(0.55, 0.65, 0.8),
      frontierSpread = frontierSpread,
      lastReward = lastReward,
      lastCacheHit = lastCacheHit
    )
  }

  private val busyState = frontierState(
    completedEvaluations = 12,
    frontierSize = 4,
    duplicateCount = 3,
    frontierSpread = 0.42,
    lastReward = 1.1,
    lastCacheHit = true
  )

  "NeuralSplitPolicy" should "produce a valid probability distribution" in {
    val policy = new NeuralSplitPolicy(
      random = new Random(1),
      hiddenSize = 12,
      temperature = 1.0,
      gradientClip = 5.0
    )

    val scored = policy.score(width, context, existingOutputs, busyState)
    scored.probabilities.length shouldBe context.actionCount
    scored.probabilities.sum shouldBe 1.0 +- 1e-9
    scored.probabilities.foreach { p =>
      p should be >= 0.0
      p should be <= 1.0
    }
  }

  it should "support alternative non-default architecture modes" in {
    val policy = new NeuralSplitPolicy(
      random = new Random(13),
      hiddenSize = 12,
      temperature = 1.0,
      gradientClip = 5.0,
      architecture = NeuralArchitectureConfig(
        dagSummaryMode = "attention-weighted",
        actionContextMode = "mean-residual"
      )
    )

    val scored = policy.score(width, context, existingOutputs, busyState)
    scored.probabilities.length shouldBe context.actionCount
    scored.probabilities.sum shouldBe 1.0 +- 1e-9
    policy.architectureConfig.normalizedDagSummaryMode shouldBe "attention-weighted"
    policy.architectureConfig.normalizedActionContextMode shouldBe "mean-residual"
    scored.cache.dagSummaryMode shouldBe "attention-weighted"
    scored.cache.actionContextMode shouldBe "mean-residual"
    scored.cache.dagSummaryWeights.length shouldBe scored.cache.dagSignatures.length
  }

  it should "condition its outputs on frontier state" in {
    val policy = new NeuralSplitPolicy(
      random = new Random(101),
      hiddenSize = 12,
      temperature = 1.0,
      gradientClip = 5.0
    )

    val cold = policy.score(width, context, existingOutputs, SearchPolicyState.empty)
    val busy = policy.score(width, context, existingOutputs, busyState)

    (cold.cache.stateInput.toVector should not).equal(busy.cache.stateInput.toVector)
    math.abs(cold.centeredValueEstimate - busy.centeredValueEstimate) should be > 1e-9
  }

  it should "reject inconsistent existingOutputs lengths" in {
    val policy = new NeuralSplitPolicy(
      random = new Random(2),
      hiddenSize = 12,
      temperature = 1.0,
      gradientClip = 5.0
    )

    an[IllegalArgumentException] should be thrownBy {
      policy.score(width, context, topology.outputs.take(context.outputIndex + 1), busyState)
    }
  }

  it should "increase the probability of rewarded actions after updates" in {
    val policy = new NeuralSplitPolicy(
      random = new Random(7),
      hiddenSize = 10,
      temperature = 1.0,
      gradientClip = 5.0
    )

    val initial = policy.score(width, context, existingOutputs, busyState)
    val targetAction = initial.probabilities.zipWithIndex.minBy(_._1)._2
    val before = initial.probabilities(targetAction)

    for (_ <- 0 until 40) {
      val scored = policy.score(width, context, existingOutputs, busyState)
      val step = SampledDecision(
        context = context,
        actionIndex = targetAction,
        probabilities = scored.probabilities,
        evaluations = scored.evaluations,
        cache = scored.cache
      )
      policy.update(
        trace = Seq(step),
        reward = 1.0,
        baseline = 0.0,
        learningRate = 0.05
      )
    }

    val after = policy.score(width, context, existingOutputs, busyState).probabilities(targetAction)
    after should be > before
  }

  it should "move the critic estimate toward the centered reward" in {
    val policy = new NeuralSplitPolicy(
      random = new Random(11),
      hiddenSize = 10,
      temperature = 1.0,
      gradientClip = 5.0
    )

    val reward = 0.75
    val baseline = 0.1
    val target = reward - baseline
    val before = policy.score(width, context, existingOutputs, busyState).centeredValueEstimate

    for (_ <- 0 until 50) {
      val scored = policy.score(width, context, existingOutputs, busyState)
      val step = SampledDecision(
        context = context,
        actionIndex = 0,
        probabilities = scored.probabilities,
        evaluations = scored.evaluations,
        cache = scored.cache
      )
      policy.update(
        trace = Seq(step),
        reward = reward,
        baseline = baseline,
        learningRate = 0.03
      )
    }

    val after = policy.score(width, context, existingOutputs, busyState).centeredValueEstimate
    math.abs(after - target) should be < math.abs(before - target)
  }

  it should "round-trip through JSON checkpoints including optimizer state" in {
    val policy = new NeuralSplitPolicy(
      random = new Random(29),
      hiddenSize = 8,
      temperature = 0.9,
      gradientClip = 4.0,
      architecture = NeuralArchitectureConfig(
        dagSummaryMode = "attention-weighted",
        actionContextMode = "mean-residual"
      )
    )

    for (_ <- 0 until 6) {
      val scored = policy.score(width, context, existingOutputs, busyState)
      val step = SampledDecision(
        context = context,
        actionIndex = 0,
        probabilities = scored.probabilities,
        evaluations = scored.evaluations,
        cache = scored.cache
      )
      policy.update(
        trace = Seq(step),
        reward = 0.8,
        baseline = 0.1,
        learningRate = 0.02
      )
    }

    val before = policy.score(width, context, existingOutputs, busyState)
    val dir = os.temp.dir(prefix = "neural_policy_roundtrip_")
    val path = policy.write(dir / "policy.json")
    val loaded = NeuralSplitPolicy.load(path, new Random(123))
    val after = loaded.score(width, context, existingOutputs, busyState)

    loaded.hiddenSizeValue shouldBe 8
    loaded.temperatureValue shouldBe 0.9 +- 1e-9
    loaded.gradientClipValue shouldBe 4.0 +- 1e-9
    loaded.architectureConfig.normalizedDagSummaryMode shouldBe "attention-weighted"
    loaded.architectureConfig.normalizedActionContextMode shouldBe "mean-residual"
    loaded.optimizerStepCount shouldBe policy.optimizerStepCount
    loaded.exportParameters().toVector shouldBe policy.exportParameters().toVector
    after.probabilities shouldBe before.probabilities
    after.centeredValueEstimate shouldBe before.centeredValueEstimate +- 1e-9
  }

  it should "match finite-difference gradients for the surrogate loss" in {
    val epsilon = 1e-6
    val tolerance = 1e-4
    val scenarios = Vector(
      (
        topology.outputs.take(4),
        DecisionContext(outputIndex = 4, segmentLow = 0, segmentHigh = 4, root = true),
        frontierState(8, 3, 1, 0.28, 0.9, lastCacheHit = false)
      ),
      (
        topology.outputs.take(5),
        DecisionContext(outputIndex = 5, segmentLow = 2, segmentHigh = 5, root = false),
        frontierState(14, 5, 4, 0.51, 1.35, lastCacheHit = true)
      )
    )

    scenarios.foreach { case (outputs, decisionContext, searchState) =>
      val policy = new NeuralSplitPolicy(
        random = new Random(19 + decisionContext.outputIndex),
        hiddenSize = 4,
        temperature = 0.9,
        gradientClip = 5.0
      )

      val scored = policy.score(width, decisionContext, outputs, searchState)
      val actionIndex = scored.probabilities.zipWithIndex.minBy(_._1)._2
      val step = SampledDecision(
        context = decisionContext,
        actionIndex = actionIndex,
        probabilities = scored.probabilities,
        evaluations = scored.evaluations,
        cache = scored.cache
      )
      val centeredReward = step.centeredValueEstimate + 0.35
      val frozenAdvantage = centeredReward - step.centeredValueEstimate

      val analytic = policy.analyticGradientVector(step, centeredReward)
      val baseParameters = policy.exportParameters()
      analytic.length shouldBe policy.parameterCount
      analytic.length shouldBe baseParameters.length

      var maxError = 0.0
      var index = 0
      while (index < baseParameters.length) {
        val plus = baseParameters.clone()
        plus(index) += epsilon
        policy.importParameters(plus)
        val lossPlus = policy.surrogateLossForSelection(
          width = width,
          context = decisionContext,
          existingOutputs = outputs,
          selectedActionIndex = actionIndex,
          centeredReward = centeredReward,
          frozenAdvantage = frozenAdvantage,
          searchState = searchState
        )

        val minus = baseParameters.clone()
        minus(index) -= epsilon
        policy.importParameters(minus)
        val lossMinus = policy.surrogateLossForSelection(
          width = width,
          context = decisionContext,
          existingOutputs = outputs,
          selectedActionIndex = actionIndex,
          centeredReward = centeredReward,
          frozenAdvantage = frozenAdvantage,
          searchState = searchState
        )

        val numerical = (lossPlus - lossMinus) / (2.0 * epsilon)
        maxError = math.max(maxError, math.abs(numerical - analytic(index)))
        index += 1
      }

      policy.importParameters(baseParameters)
      maxError should be < tolerance
    }
  }

  it should "write checkpoints as JSON" in {
    val policy = new NeuralSplitPolicy(
      random = new Random(3),
      hiddenSize = 8,
      temperature = 1.0,
      gradientClip = 5.0
    )

    val dir = os.temp.dir(prefix = "neural_policy_spec_")
    val path = policy.write(dir / "policy.json")
    os.exists(path) shouldBe true

    val json = ujson.read(os.read(path))
    json("policy").str shouldBe "frontier-conditioned-dag-pointer-actor-critic"
    json("hidden_size").num.toInt shouldBe 8
    json("action_feature_dimension").num.toInt shouldBe ActionFeatureEncoder.dimension
    json("state_feature_dimension").num.toInt shouldBe StateFeatureEncoder.dimension
    json("leaf_feature_dimension").num.toInt shouldBe LeafFeatureEncoder.dimension
    json("node_feature_dimension").num.toInt shouldBe NodeFeatureEncoder.dimension
    json("actor_head").obj.contains("bias") shouldBe false
    json("tree_encoder").obj.contains("leaf_encoder") shouldBe true
    json("architecture")("dag_summary_mode").str shouldBe "usage-weighted"
    json("optimizer").obj.contains("leaf_encoder") shouldBe true
  }
}
