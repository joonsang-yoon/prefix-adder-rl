package PrefixDeepRL

import PrefixAdderLib.{Leaf, Node, PrefixTree}
import PrefixRLCore.{DecisionContext, SearchPolicyState, SplitDecision, SplitPolicy}
import PrefixUtils.JsonSupport

import scala.collection.mutable
import scala.util.Random

final case class PolicyTrainingStats(
  advantage:           Double,
  averageEntropy:      Double,
  averageValue:        Double,
  valueLoss:           Double,
  gradientNorm:        Double,
  clippedGradientNorm: Double
) {
  def toJson: ujson.Value = ujson.Obj(
    "advantage" -> advantage,
    "average_entropy" -> averageEntropy,
    "average_value" -> averageValue,
    "value_loss" -> valueLoss,
    "gradient_norm" -> gradientNorm,
    "clipped_gradient_norm" -> clippedGradientNorm
  )
}

final case class NeuralArchitectureConfig(
  dagSummaryMode:    String = NeuralArchitectureConfig.DefaultDagSummaryMode,
  actionContextMode: String = NeuralArchitectureConfig.DefaultActionContextMode
) {
  val normalizedDagSummaryMode:    String = dagSummaryMode.trim.toLowerCase
  val normalizedActionContextMode: String = actionContextMode.trim.toLowerCase

  require(
    NeuralArchitectureConfig.SupportedDagSummaryModes.contains(normalizedDagSummaryMode),
    s"Unsupported dagSummaryMode '${dagSummaryMode}'. Expected one of: ${NeuralArchitectureConfig.SupportedDagSummaryModes.mkString(", ")}"
  )
  require(
    NeuralArchitectureConfig.SupportedActionContextModes.contains(normalizedActionContextMode),
    s"Unsupported actionContextMode '${actionContextMode}'. Expected one of: ${NeuralArchitectureConfig.SupportedActionContextModes.mkString(", ")}"
  )

  def toJson: ujson.Value = ujson.Obj(
    "dag_summary_mode" -> normalizedDagSummaryMode,
    "action_context_mode" -> normalizedActionContextMode
  )
}

object NeuralArchitectureConfig {
  val DefaultDagSummaryMode:    String = "usage-weighted"
  val DefaultActionContextMode: String = "self-attention"

  val SupportedDagSummaryModes: Vector[String] = Vector(
    DefaultDagSummaryMode,
    "attention-weighted"
  )

  val SupportedActionContextModes: Vector[String] = Vector(
    DefaultActionContextMode,
    "mean-residual"
  )

  val default: NeuralArchitectureConfig = NeuralArchitectureConfig()

  def fromJson(json: ujson.Value): NeuralArchitectureConfig = NeuralArchitectureConfig(
    dagSummaryMode = json.obj.get("dag_summary_mode").map(JsonSupport.readString).getOrElse(DefaultDagSummaryMode),
    actionContextMode =
      json.obj.get("action_context_mode").map(JsonSupport.readString).getOrElse(DefaultActionContextMode)
  )
}

object ActionFeatureEncoder {
  val dimension: Int = 22

  def encode(
    width:           Int,
    context:         DecisionContext,
    actionIndex:     Int,
    existingOutputs: IndexedSeq[PrefixTree]
  ): Array[Double] = {
    require(width >= 1, s"width must be >= 1, got ${width}")

    val widthNorm = math.max(1.0, width.toDouble)
    val maxIndex = math.max(1.0, (width - 1).toDouble)
    val segmentLength = context.segmentHigh - context.segmentLow + 1
    val actionCount = context.actionCount
    val absoluteSplit = context.absoluteSplit(actionIndex)
    val leftSpan = absoluteSplit - context.segmentLow + 1
    val rightSpan = context.segmentHigh - absoluteSplit

    val reusedPrefix =
      if (context.root && absoluteSplit >= 0 && absoluteSplit < existingOutputs.length)
        Some(existingOutputs(absoluteSplit))
      else None

    val reusedDepth = reusedPrefix.map(_.depth.toDouble).getOrElse(0.0)
    val reusedInternalNodes = reusedPrefix.map(_.internalNodeCount.toDouble).getOrElse(0.0)
    val reusedLeaves = reusedPrefix.map(_.leafCount.toDouble).getOrElse(0.0)

    val segmentLengthNorm = segmentLength.toDouble / widthNorm
    val actionIndexNorm = actionIndex.toDouble / math.max(1.0, (actionCount - 1).toDouble)
    val actionCountNorm = actionCount.toDouble / widthNorm
    val splitNorm = absoluteSplit.toDouble / maxIndex
    val leftFraction = leftSpan.toDouble / segmentLength.toDouble
    val rightFraction = rightSpan.toDouble / segmentLength.toDouble
    val balance = 1.0 - math.abs(leftSpan - rightSpan).toDouble / segmentLength.toDouble
    val centeredSplit = ((absoluteSplit.toDouble + 0.5) / widthNorm) * 2.0 - 1.0
    val reusedDepthNorm = reusedDepth / maxIndex
    val reusedInternalNorm = reusedInternalNodes / math.max(1.0, (width - 1).toDouble)
    val reusedLeafNorm = reusedLeaves / widthNorm
    val logLeftSpan = math.log1p(leftSpan.toDouble) / math.log1p(widthNorm)
    val logRightSpan = math.log1p(rightSpan.toDouble) / math.log1p(widthNorm)

    Array[Double](
      1.0,
      context.outputIndex.toDouble / maxIndex,
      context.segmentLow.toDouble / maxIndex,
      context.segmentHigh.toDouble / maxIndex,
      segmentLengthNorm,
      if (context.root) 1.0 else 0.0,
      actionIndexNorm,
      actionCountNorm,
      splitNorm,
      leftFraction,
      rightFraction,
      balance,
      if (rightSpan == 1) 1.0 else 0.0,
      if (leftSpan == 1) 1.0 else 0.0,
      reusedDepthNorm,
      reusedInternalNorm,
      reusedLeafNorm,
      logRightSpan,
      logLeftSpan,
      centeredSplit,
      (if (context.root) 1.0 else 0.0) * (context.outputIndex.toDouble / maxIndex),
      reusedDepthNorm * rightFraction
    )
  }
}

object StateFeatureEncoder {
  val dimension: Int = 41

  def encode(
    width:           Int,
    context:         DecisionContext,
    existingOutputs: IndexedSeq[PrefixTree],
    searchState:     SearchPolicyState
  ): Array[Double] = {
    require(width >= 1, s"width must be >= 1, got ${width}")

    val widthNorm = math.max(1.0, width.toDouble)
    val maxIndex = math.max(1.0, (width - 1).toDouble)
    val internalNorm = math.max(1.0, (width - 1).toDouble)
    val segmentLength = context.segmentHigh - context.segmentLow + 1
    val segmentMid = 0.5 * (context.segmentLow.toDouble + context.segmentHigh.toDouble)
    val actionDensity = context.actionCount.toDouble / segmentLength.toDouble

    val depths = existingOutputs.map(_.depth.toDouble)
    val internalNodes = existingOutputs.map(_.internalNodeCount.toDouble)
    val leaves = existingOutputs.map(_.leafCount.toDouble)

    val meanDepth = if (depths.isEmpty) 0.0 else depths.sum / depths.length.toDouble
    val maxDepth = if (depths.isEmpty) 0.0 else depths.max
    val meanInternal = if (internalNodes.isEmpty) 0.0 else internalNodes.sum / internalNodes.length.toDouble
    val maxInternal = if (internalNodes.isEmpty) 0.0 else internalNodes.max
    val meanLeaves = if (leaves.isEmpty) 0.0 else leaves.sum / leaves.length.toDouble
    val maxLeaves = if (leaves.isEmpty) 0.0 else leaves.max

    val totalInternal = existingOutputs.map(_.internalNodeCount.toDouble).sum
    val uniqueInternal = countUniqueInternalNodes(existingOutputs).toDouble
    val reuseRatio =
      if (totalInternal == 0.0) 1.0
      else 1.0 - uniqueInternal / totalInternal

    val lastOutput = existingOutputs.lastOption
    val lastDepth = lastOutput.map(_.depth.toDouble).getOrElse(0.0)
    val lastInternal = lastOutput.map(_.internalNodeCount.toDouble).getOrElse(0.0)
    val lastLeaves = lastOutput.map(_.leafCount.toDouble).getOrElse(0.0)

    Array[Double](
      1.0,
      context.outputIndex.toDouble / maxIndex,
      context.segmentLow.toDouble / maxIndex,
      context.segmentHigh.toDouble / maxIndex,
      segmentLength.toDouble / widthNorm,
      if (context.root) 1.0 else 0.0,
      context.actionCount.toDouble / widthNorm,
      existingOutputs.length.toDouble / widthNorm,
      meanDepth / maxIndex,
      maxDepth / maxIndex,
      meanInternal / internalNorm,
      maxInternal / internalNorm,
      meanLeaves / widthNorm,
      maxLeaves / widthNorm,
      reuseRatio,
      lastDepth / maxIndex,
      lastInternal / internalNorm,
      lastLeaves / widthNorm,
      ((segmentMid + 0.5) / widthNorm) * 2.0 - 1.0,
      actionDensity,
      if (searchState.hasObservations) 1.0 else 0.0,
      if (searchState.hasFrontier) 1.0 else 0.0,
      boundedCount(searchState.completedEvaluations),
      boundedCount(searchState.frontierSize),
      boundedCount(searchState.cacheSize),
      boundedCount(searchState.duplicateCount),
      searchState.duplicateRate,
      searchState.frontierFraction,
      searchState.uniqueFraction,
      searchState.normalizedBest.power,
      searchState.normalizedBest.delay,
      searchState.normalizedBest.area,
      searchState.normalizedMean.power,
      searchState.normalizedMean.delay,
      searchState.normalizedMean.area,
      searchState.normalizedWorst.power,
      searchState.normalizedWorst.delay,
      searchState.normalizedWorst.area,
      math.tanh(searchState.frontierSpread),
      squashReward(searchState.lastReward),
      if (searchState.lastCacheHit) 1.0 else 0.0
    )
  }

  private def boundedCount(count: Int): Double = {
    val positive = math.max(0, count).toDouble
    positive / (positive + 4.0)
  }

  private def squashReward(reward: Double): Double = math.tanh(reward / 4.0)

  private def countUniqueInternalNodes(existingOutputs: IndexedSeq[PrefixTree]): Int = {
    val unique = mutable.HashSet.empty[String]
    existingOutputs.foreach(tree => collectInternalNodeSignatures(tree, unique))
    unique.size
  }

  private def collectInternalNodeSignatures(tree: PrefixTree, acc: mutable.Set[String]): Unit = tree match {
    case Leaf(_) =>
    case node @ Node(left, right) =>
      acc += node.signature
      collectInternalNodeSignatures(left, acc)
      collectInternalNodeSignatures(right, acc)
  }
}

object LeafFeatureEncoder {
  val dimension: Int = 5

  def encode(width: Int, index: Int): Array[Double] = {
    require(width >= 1, s"width must be >= 1, got ${width}")

    val widthNorm = math.max(1.0, width.toDouble)
    val maxIndex = math.max(1.0, (width - 1).toDouble)
    val indexNorm = index.toDouble / maxIndex
    val centered = ((index.toDouble + 0.5) / widthNorm) * 2.0 - 1.0

    Array[Double](
      1.0,
      indexNorm,
      centered,
      if (index == 0) 1.0 else 0.0,
      if (index == width - 1) 1.0 else 0.0
    )
  }
}

object NodeFeatureEncoder {
  val dimension: Int = 10

  def encode(width: Int, node: Node): Array[Double] = {
    require(width >= 1, s"width must be >= 1, got ${width}")

    val widthNorm = math.max(1.0, width.toDouble)
    val maxIndex = math.max(1.0, (width - 1).toDouble)
    val span = node.high - node.low + 1
    val leftSpan = node.left.high - node.left.low + 1
    val rightSpan = node.right.high - node.right.low + 1
    val midpoint = 0.5 * (node.low.toDouble + node.high.toDouble)
    val balance = 1.0 - math.abs(leftSpan - rightSpan).toDouble / span.toDouble

    Array[Double](
      1.0,
      node.low.toDouble / maxIndex,
      node.high.toDouble / maxIndex,
      span.toDouble / widthNorm,
      ((midpoint + 0.5) / widthNorm) * 2.0 - 1.0,
      leftSpan.toDouble / span.toDouble,
      rightSpan.toDouble / span.toDouble,
      balance,
      node.left.depth.toDouble / maxIndex,
      node.right.depth.toDouble / maxIndex
    )
  }
}

final case class TreeEmbeddingCache(
  signature:      String,
  input:          Array[Double],
  embedding:      Array[Double],
  isLeaf:         Boolean,
  leftSignature:  Option[String],
  rightSignature: Option[String]
)

final case class EncodedTopology(
  outputSignatures: Vector[String],
  dagSignatures:    Vector[String],
  dagWeights:       Array[Double],
  treeCaches:       Map[String, TreeEmbeddingCache],
  treeCacheOrder:   Vector[String]
)

final case class DecisionForwardCache(
  stateInput:             Array[Double],
  topologyQuery:          Array[Double],
  topologySummary:        Array[Double],
  dagSummary:             Array[Double],
  outputAttentionWeights: Array[Double],
  outputSignatures:       Vector[String],
  dagSignatures:          Vector[String],
  dagSummaryWeights:      Array[Double],
  dagSummaryMode:         String,
  treeCaches:             Map[String, TreeEmbeddingCache],
  treeCacheOrder:         Vector[String],
  actionContextMode:      String,
  actionAttentionWeights: Array[Array[Double]],
  actionSummary:          Array[Double],
  queryInput:             Array[Double],
  query:                  Array[Double],
  centeredValueEstimate:  Double
)

final case class CandidateEvaluation(
  actionIndex:         Int,
  absoluteSplit:       Int,
  logit:               Double,
  actionInput:         Array[Double],
  actionEmbedding:     Array[Double],
  contextualEmbedding: Array[Double]
)

final case class ScoredDecision(
  context:       DecisionContext,
  evaluations:   Vector[CandidateEvaluation],
  probabilities: Vector[Double],
  cache:         DecisionForwardCache
) {
  require(evaluations.nonEmpty, s"scored decision must contain at least one candidate: ${context}")
  require(
    evaluations.length == probabilities.length,
    s"candidate/probability mismatch for ${context}: ${evaluations.length} != ${probabilities.length}"
  )

  lazy val entropy:          Double = probabilities.filter(_ > 0.0).map(p => -p * math.log(p)).sum
  def centeredValueEstimate: Double = cache.centeredValueEstimate

  def sample(random: Random): SampledDecision = {
    val draw = random.nextDouble()
    var cumulative = 0.0
    var chosen = 0

    while (chosen < probabilities.length - 1) {
      cumulative += probabilities(chosen)
      if (draw < cumulative) {
        return SampledDecision(
          context = context,
          actionIndex = chosen,
          probabilities = probabilities,
          evaluations = evaluations,
          cache = cache
        )
      }
      chosen += 1
    }

    SampledDecision(
      context = context,
      actionIndex = probabilities.length - 1,
      probabilities = probabilities,
      evaluations = evaluations,
      cache = cache
    )
  }
}

final case class SampledDecision(
  context:       DecisionContext,
  actionIndex:   Int,
  probabilities: Vector[Double],
  evaluations:   Vector[CandidateEvaluation],
  cache:         DecisionForwardCache
) extends SplitDecision {
  lazy val entropy:          Double = probabilities.filter(_ > 0.0).map(p => -p * math.log(p)).sum
  def centeredValueEstimate: Double = cache.centeredValueEstimate
}

private[PrefixDeepRL] final class NetworkGradients(
  actionInputSize: Int,
  stateInputSize:  Int,
  leafInputSize:   Int,
  nodeInputSize:   Int,
  hiddenSize:      Int
) {
  val leafEncoder:              Array[Array[Double]] = Array.fill(hiddenSize, leafInputSize)(0.0)
  val leafEncoderBias:          Array[Double] = Array.fill(hiddenSize)(0.0)
  val nodeEncoder:              Array[Array[Double]] = Array.fill(hiddenSize, nodeInputSize)(0.0)
  val nodeEncoderBias:          Array[Double] = Array.fill(hiddenSize)(0.0)
  val actionEncoder:            Array[Array[Double]] = Array.fill(hiddenSize, actionInputSize)(0.0)
  val actionEncoderBias:        Array[Double] = Array.fill(hiddenSize)(0.0)
  val topologyQueryEncoder:     Array[Array[Double]] = Array.fill(hiddenSize, stateInputSize)(0.0)
  val topologyQueryEncoderBias: Array[Double] = Array.fill(hiddenSize)(0.0)
  val queryEncoder:             Array[Array[Double]] = Array.fill(hiddenSize, stateInputSize + 3 * hiddenSize)(0.0)
  val queryEncoderBias:         Array[Double] = Array.fill(hiddenSize)(0.0)
  val actorHead:                Array[Double] = Array.fill(hiddenSize)(0.0)
  val valueHead:                Array[Double] = Array.fill(hiddenSize)(0.0)
  var valueBias:                Double = 0.0

  def l2Norm: Double = {
    var sumSquares = 0.0

    sumSquares += sumSquares2d(leafEncoder)
    sumSquares += sumSquares2d(nodeEncoder)
    sumSquares += sumSquares2d(actionEncoder)
    sumSquares += sumSquares2d(topologyQueryEncoder)
    sumSquares += sumSquares2d(queryEncoder)
    sumSquares += sumSquares1d(leafEncoderBias)
    sumSquares += sumSquares1d(nodeEncoderBias)
    sumSquares += sumSquares1d(actionEncoderBias)
    sumSquares += sumSquares1d(topologyQueryEncoderBias)
    sumSquares += sumSquares1d(queryEncoderBias)
    sumSquares += sumSquares1d(actorHead)
    sumSquares += sumSquares1d(valueHead)
    sumSquares += valueBias * valueBias

    math.sqrt(sumSquares)
  }

  def clipInPlace(maxNorm: Double): Double = {
    val norm = l2Norm
    if (norm <= maxNorm || norm == 0.0) {
      norm
    } else {
      val scale = maxNorm / norm
      scale2dInPlace(leafEncoder, scale)
      scale2dInPlace(nodeEncoder, scale)
      scale2dInPlace(actionEncoder, scale)
      scale2dInPlace(topologyQueryEncoder, scale)
      scale2dInPlace(queryEncoder, scale)
      scale1dInPlace(leafEncoderBias, scale)
      scale1dInPlace(nodeEncoderBias, scale)
      scale1dInPlace(actionEncoderBias, scale)
      scale1dInPlace(topologyQueryEncoderBias, scale)
      scale1dInPlace(queryEncoderBias, scale)
      scale1dInPlace(actorHead, scale)
      scale1dInPlace(valueHead, scale)
      valueBias *= scale
      maxNorm
    }
  }

  private def sumSquares1d(values: Array[Double]): Double = {
    var sum = 0.0
    var idx = 0
    while (idx < values.length) {
      val v = values(idx)
      sum += v * v
      idx += 1
    }
    sum
  }

  private def sumSquares2d(values: Array[Array[Double]]): Double = {
    var sum = 0.0
    var row = 0
    while (row < values.length) {
      sum += sumSquares1d(values(row))
      row += 1
    }
    sum
  }

  private def scale1dInPlace(values: Array[Double], scale: Double): Unit = {
    var idx = 0
    while (idx < values.length) {
      values(idx) *= scale
      idx += 1
    }
  }

  private def scale2dInPlace(values: Array[Array[Double]], scale: Double): Unit = {
    var row = 0
    while (row < values.length) {
      scale1dInPlace(values(row), scale)
      row += 1
    }
  }
}

object NeuralSplitPolicy {
  def load(path: os.Path, random: Random): NeuralSplitPolicy =
    fromJson(ujson.read(os.read(path)), random)

  def fromJson(json: ujson.Value, random: Random): NeuralSplitPolicy = {
    val hiddenSize = JsonSupport.readInt(json("hidden_size"))
    val temperature = JsonSupport.readDouble(json("temperature"))
    val gradientClip = JsonSupport.readDouble(json("gradient_clip"))
    val architecture = json.obj.get("architecture") match {
      case Some(value) => NeuralArchitectureConfig.fromJson(value)
      case None        => NeuralArchitectureConfig.default
    }

    val policy = new NeuralSplitPolicy(
      random = random,
      hiddenSize = hiddenSize,
      temperature = temperature,
      gradientClip = gradientClip,
      architecture = architecture
    )
    policy.loadFromJson(json)
    policy
  }
}

final class NeuralSplitPolicy(
  random:       Random,
  hiddenSize:   Int,
  temperature:  Double,
  gradientClip: Double,
  architecture: NeuralArchitectureConfig = NeuralArchitectureConfig.default
) extends SplitPolicy[SampledDecision] {
  require(hiddenSize >= 1, s"hiddenSize must be >= 1, got ${hiddenSize}")
  require(temperature > 0.0, s"temperature must be > 0, got ${temperature}")
  require(gradientClip > 0.0, s"gradientClip must be > 0, got ${gradientClip}")

  private val actionInputSize = ActionFeatureEncoder.dimension
  private val stateInputSize = StateFeatureEncoder.dimension
  private val leafInputSize = LeafFeatureEncoder.dimension
  private val nodeFeatureSize = NodeFeatureEncoder.dimension
  private val nodeInputSize = 2 * hiddenSize + nodeFeatureSize
  private val queryInputSize = stateInputSize + 3 * hiddenSize
  private val interactionScale = 1.0 / math.sqrt(hiddenSize.toDouble)
  private val dagSummaryMode = architecture.normalizedDagSummaryMode
  private val actionContextMode = architecture.normalizedActionContextMode

  private val leafEncoder:              Array[Array[Double]] = Array.fill(hiddenSize, leafInputSize)(0.0)
  private val leafEncoderBias:          Array[Double] = Array.fill(hiddenSize)(0.0)
  private val nodeEncoder:              Array[Array[Double]] = Array.fill(hiddenSize, nodeInputSize)(0.0)
  private val nodeEncoderBias:          Array[Double] = Array.fill(hiddenSize)(0.0)
  private val actionEncoder:            Array[Array[Double]] = Array.fill(hiddenSize, actionInputSize)(0.0)
  private val actionEncoderBias:        Array[Double] = Array.fill(hiddenSize)(0.0)
  private val topologyQueryEncoder:     Array[Array[Double]] = Array.fill(hiddenSize, stateInputSize)(0.0)
  private val topologyQueryEncoderBias: Array[Double] = Array.fill(hiddenSize)(0.0)
  private val queryEncoder:             Array[Array[Double]] = Array.fill(hiddenSize, queryInputSize)(0.0)
  private val queryEncoderBias:         Array[Double] = Array.fill(hiddenSize)(0.0)
  private val actorHead:                Array[Double] = Array.fill(hiddenSize)(0.0)
  private val valueHead:                Array[Double] = Array.fill(hiddenSize)(0.0)
  private var valueBias:                Double = 0.0

  private val mLeafEncoder:              Array[Array[Double]] = Array.fill(hiddenSize, leafInputSize)(0.0)
  private val vLeafEncoder:              Array[Array[Double]] = Array.fill(hiddenSize, leafInputSize)(0.0)
  private val mLeafEncoderBias:          Array[Double] = Array.fill(hiddenSize)(0.0)
  private val vLeafEncoderBias:          Array[Double] = Array.fill(hiddenSize)(0.0)
  private val mNodeEncoder:              Array[Array[Double]] = Array.fill(hiddenSize, nodeInputSize)(0.0)
  private val vNodeEncoder:              Array[Array[Double]] = Array.fill(hiddenSize, nodeInputSize)(0.0)
  private val mNodeEncoderBias:          Array[Double] = Array.fill(hiddenSize)(0.0)
  private val vNodeEncoderBias:          Array[Double] = Array.fill(hiddenSize)(0.0)
  private val mActionEncoder:            Array[Array[Double]] = Array.fill(hiddenSize, actionInputSize)(0.0)
  private val vActionEncoder:            Array[Array[Double]] = Array.fill(hiddenSize, actionInputSize)(0.0)
  private val mActionEncoderBias:        Array[Double] = Array.fill(hiddenSize)(0.0)
  private val vActionEncoderBias:        Array[Double] = Array.fill(hiddenSize)(0.0)
  private val mTopologyQueryEncoder:     Array[Array[Double]] = Array.fill(hiddenSize, stateInputSize)(0.0)
  private val vTopologyQueryEncoder:     Array[Array[Double]] = Array.fill(hiddenSize, stateInputSize)(0.0)
  private val mTopologyQueryEncoderBias: Array[Double] = Array.fill(hiddenSize)(0.0)
  private val vTopologyQueryEncoderBias: Array[Double] = Array.fill(hiddenSize)(0.0)
  private val mQueryEncoder:             Array[Array[Double]] = Array.fill(hiddenSize, queryInputSize)(0.0)
  private val vQueryEncoder:             Array[Array[Double]] = Array.fill(hiddenSize, queryInputSize)(0.0)
  private val mQueryEncoderBias:         Array[Double] = Array.fill(hiddenSize)(0.0)
  private val vQueryEncoderBias:         Array[Double] = Array.fill(hiddenSize)(0.0)
  private val mActorHead:                Array[Double] = Array.fill(hiddenSize)(0.0)
  private val vActorHead:                Array[Double] = Array.fill(hiddenSize)(0.0)
  private val mValueHead:                Array[Double] = Array.fill(hiddenSize)(0.0)
  private val vValueHead:                Array[Double] = Array.fill(hiddenSize)(0.0)
  private var mValueBias:                Double = 0.0
  private var vValueBias:                Double = 0.0
  private var optimizerStep:             Long = 0L

  def architectureConfig: NeuralArchitectureConfig = architecture
  def optimizerStepCount: Long = optimizerStep
  def hiddenSizeValue:    Int = hiddenSize
  def temperatureValue:   Double = temperature
  def gradientClipValue:  Double = gradientClip

  initializeParameters()

  private def requireConsistentExistingOutputs(
    context:         DecisionContext,
    existingOutputs: IndexedSeq[PrefixTree]
  ): Unit = {
    require(
      existingOutputs.length == context.outputIndex,
      s"inconsistent existingOutputs for ${context}: expected ${context.outputIndex} prior outputs, got ${existingOutputs.length}"
    )
  }

  def score(
    width:           Int,
    context:         DecisionContext,
    existingOutputs: IndexedSeq[PrefixTree],
    searchState:     SearchPolicyState
  ): ScoredDecision = {
    requireConsistentExistingOutputs(context, existingOutputs)

    val stateInput = StateFeatureEncoder.encode(width, context, existingOutputs, searchState)

    val encodedTopology = encodeExistingOutputs(width, existingOutputs)
    val outputEmbeddings =
      encodedTopology.outputSignatures.map(signature => encodedTopology.treeCaches(signature).embedding)
    val dagEmbeddings = encodedTopology.dagSignatures.map(signature => encodedTopology.treeCaches(signature).embedding)
    val topologyQuery = tanhLayer(topologyQueryEncoder, topologyQueryEncoderBias, stateInput)
    val outputAttentionWeights = softmax(
      outputEmbeddings.map(embedding => interactionScale * dot(topologyQuery, embedding)).toArray,
      temperature = 1.0
    )
    val topologySummary = weightedAverage(outputEmbeddings, outputAttentionWeights)
    val dagSummaryWeights =
      if (dagEmbeddings.isEmpty) {
        Array.empty[Double]
      } else if (dagSummaryMode == NeuralArchitectureConfig.DefaultDagSummaryMode) {
        encodedTopology.dagWeights
      } else {
        softmax(
          dagEmbeddings.map(embedding => interactionScale * dot(topologyQuery, embedding)).toArray,
          temperature = 1.0
        )
      }
    val dagSummary = if (dagEmbeddings.isEmpty) {
      Array.fill(hiddenSize)(0.0)
    } else {
      weightedAverage(dagEmbeddings, dagSummaryWeights)
    }

    val baseCandidates = (0 until context.actionCount).map { actionIndex =>
      val actionInput = ActionFeatureEncoder.encode(width, context, actionIndex, existingOutputs)
      val actionEmbedding = tanhLayer(actionEncoder, actionEncoderBias, actionInput)
      CandidateEvaluation(
        actionIndex = actionIndex,
        absoluteSplit = context.absoluteSplit(actionIndex),
        logit = 0.0,
        actionInput = actionInput,
        actionEmbedding = actionEmbedding,
        contextualEmbedding = Array.fill(hiddenSize)(0.0)
      )
    }.toVector

    val actionEmbeddings = baseCandidates.map(_.actionEmbedding)
    val (actionAttentionWeights, contextualEmbeddings) =
      if (actionContextMode == NeuralArchitectureConfig.DefaultActionContextMode) {
        val weights = computeActionAttentionWeights(actionEmbeddings)
        val contextual = actionEmbeddings.indices.map { index =>
          val attended = weightedAverage(actionEmbeddings, weights(index))
          tanhVector(add(actionEmbeddings(index), attended))
        }.toVector
        (weights, contextual)
      } else {
        val meanEmbedding = averageEmbeddings(actionEmbeddings)
        val contextual = actionEmbeddings.map(embedding => tanhVector(add(embedding, meanEmbedding))).toVector
        (Array.ofDim[Double](0, 0), contextual)
      }
    val actionSummary = averageEmbeddings(contextualEmbeddings)

    val queryInput = concat(stateInput, topologySummary, dagSummary, actionSummary)
    val query = tanhLayer(queryEncoder, queryEncoderBias, queryInput)
    val centeredValueEstimate = dot(valueHead, query) + valueBias

    val evaluations = baseCandidates.indices.map { index =>
      val contextualEmbedding = contextualEmbeddings(index)
      val linearScore = dot(actorHead, contextualEmbedding)
      val pointerScore = interactionScale * dot(query, contextualEmbedding)
      baseCandidates(index).copy(
        logit = linearScore + pointerScore,
        contextualEmbedding = contextualEmbedding
      )
    }.toVector

    val probabilities = softmax(evaluations.map(_.logit).toArray, temperature).toVector
    ScoredDecision(
      context = context,
      evaluations = evaluations,
      probabilities = probabilities,
      cache = DecisionForwardCache(
        stateInput = stateInput,
        topologyQuery = topologyQuery,
        topologySummary = topologySummary,
        dagSummary = dagSummary,
        outputAttentionWeights = outputAttentionWeights,
        outputSignatures = encodedTopology.outputSignatures,
        dagSignatures = encodedTopology.dagSignatures,
        dagSummaryWeights = dagSummaryWeights,
        dagSummaryMode = dagSummaryMode,
        treeCaches = encodedTopology.treeCaches,
        treeCacheOrder = encodedTopology.treeCacheOrder,
        actionContextMode = actionContextMode,
        actionAttentionWeights = actionAttentionWeights,
        actionSummary = actionSummary,
        queryInput = queryInput,
        query = query,
        centeredValueEstimate = centeredValueEstimate
      )
    )
  }

  def score(
    width:           Int,
    context:         DecisionContext,
    existingOutputs: IndexedSeq[PrefixTree]
  ): ScoredDecision = score(width, context, existingOutputs, SearchPolicyState.empty)

  override def sample(
    searchState:     SearchPolicyState,
    width:           Int,
    context:         DecisionContext,
    existingOutputs: IndexedSeq[PrefixTree]
  ): SampledDecision = score(width, context, existingOutputs, searchState).sample(random)

  def sample(
    width:           Int,
    context:         DecisionContext,
    existingOutputs: IndexedSeq[PrefixTree]
  ): SampledDecision = sample(SearchPolicyState.empty, width, context, existingOutputs)

  def update(
    trace:        Seq[SampledDecision],
    reward:       Double,
    baseline:     Double,
    learningRate: Double
  ): PolicyTrainingStats = {
    require(learningRate > 0.0, s"learningRate must be > 0, got ${learningRate}")

    val centeredReward = reward - baseline
    if (trace.isEmpty) {
      return PolicyTrainingStats(
        advantage = centeredReward,
        averageEntropy = 0.0,
        averageValue = 0.0,
        valueLoss = 0.0,
        gradientNorm = 0.0,
        clippedGradientNorm = 0.0
      )
    }

    val grads = new NetworkGradients(actionInputSize, stateInputSize, leafInputSize, nodeInputSize, hiddenSize)
    var entropySum = 0.0
    var advantageSum = 0.0
    var valueSum = 0.0
    var valueLossSum = 0.0

    trace.foreach { step =>
      val centeredValueEstimate = step.centeredValueEstimate
      val advantage = centeredReward - centeredValueEstimate
      val valueResidual = centeredValueEstimate - centeredReward

      entropySum += step.entropy
      advantageSum += advantage
      valueSum += centeredValueEstimate
      valueLossSum += 0.5 * valueResidual * valueResidual

      backprop(step, advantage, valueResidual, grads)
    }

    val gradientNorm = grads.l2Norm
    val clippedGradientNorm = grads.clipInPlace(gradientClip)
    applyGradients(grads, learningRate)

    PolicyTrainingStats(
      advantage = advantageSum / trace.length.toDouble,
      averageEntropy = entropySum / trace.length.toDouble,
      averageValue = valueSum / trace.length.toDouble,
      valueLoss = valueLossSum / trace.length.toDouble,
      gradientNorm = gradientNorm,
      clippedGradientNorm = clippedGradientNorm
    )
  }

  private[PrefixDeepRL] def parameterCount: Int =
    hiddenSize * leafInputSize +
      hiddenSize +
      hiddenSize * nodeInputSize +
      hiddenSize +
      hiddenSize * actionInputSize +
      hiddenSize +
      hiddenSize * stateInputSize +
      hiddenSize +
      hiddenSize * queryInputSize +
      hiddenSize +
      hiddenSize +
      hiddenSize +
      1

  private[PrefixDeepRL] def exportParameters(): Array[Double] = {
    val out = Array.ofDim[Double](parameterCount)
    var idx = 0

    idx = export2d(leafEncoder, out, idx)
    idx = export1d(leafEncoderBias, out, idx)
    idx = export2d(nodeEncoder, out, idx)
    idx = export1d(nodeEncoderBias, out, idx)
    idx = export2d(actionEncoder, out, idx)
    idx = export1d(actionEncoderBias, out, idx)
    idx = export2d(topologyQueryEncoder, out, idx)
    idx = export1d(topologyQueryEncoderBias, out, idx)
    idx = export2d(queryEncoder, out, idx)
    idx = export1d(queryEncoderBias, out, idx)
    idx = export1d(actorHead, out, idx)
    idx = export1d(valueHead, out, idx)
    out(idx) = valueBias
    out
  }

  private[PrefixDeepRL] def importParameters(values: Array[Double]): Unit = {
    require(values.length == parameterCount, s"parameter length mismatch: ${values.length} != ${parameterCount}")

    var idx = 0
    idx = import2d(values, idx, leafEncoder)
    idx = import1d(values, idx, leafEncoderBias)
    idx = import2d(values, idx, nodeEncoder)
    idx = import1d(values, idx, nodeEncoderBias)
    idx = import2d(values, idx, actionEncoder)
    idx = import1d(values, idx, actionEncoderBias)
    idx = import2d(values, idx, topologyQueryEncoder)
    idx = import1d(values, idx, topologyQueryEncoderBias)
    idx = import2d(values, idx, queryEncoder)
    idx = import1d(values, idx, queryEncoderBias)
    idx = import1d(values, idx, actorHead)
    idx = import1d(values, idx, valueHead)
    valueBias = values(idx)
  }

  private[PrefixDeepRL] def analyticGradientVector(
    step:           SampledDecision,
    centeredReward: Double
  ): Array[Double] = {
    val grads = new NetworkGradients(actionInputSize, stateInputSize, leafInputSize, nodeInputSize, hiddenSize)
    val advantage = centeredReward - step.centeredValueEstimate
    val valueResidual = step.centeredValueEstimate - centeredReward
    backprop(step, advantage, valueResidual, grads)
    flattenGradients(grads)
  }

  private[PrefixDeepRL] def surrogateLossForSelection(
    width:               Int,
    context:             DecisionContext,
    existingOutputs:     IndexedSeq[PrefixTree],
    selectedActionIndex: Int,
    centeredReward:      Double,
    frozenAdvantage:     Double,
    searchState:         SearchPolicyState = SearchPolicyState.empty
  ): Double = {
    val scored = score(width, context, existingOutputs, searchState)
    require(
      selectedActionIndex >= 0 && selectedActionIndex < scored.probabilities.length,
      s"selectedActionIndex ${selectedActionIndex} out of range for ${scored.probabilities.length} actions"
    )

    val probability = math.max(1e-12, scored.probabilities(selectedActionIndex))
    val policyLoss = -frozenAdvantage * math.log(probability)
    val valueResidual = scored.centeredValueEstimate - centeredReward
    val valueLoss = 0.5 * valueResidual * valueResidual
    policyLoss + valueLoss
  }

  def write(path: os.Path): os.Path = {
    os.makeDir.all(path / os.up)
    os.write.over(path, ujson.write(toJson, indent = 2))
    path
  }

  def toJson: ujson.Value = ujson.Obj(
    "policy" -> "frontier-conditioned-dag-pointer-actor-critic",
    "architecture" -> architecture.toJson,
    "action_feature_dimension" -> actionInputSize,
    "state_feature_dimension" -> stateInputSize,
    "leaf_feature_dimension" -> leafInputSize,
    "node_feature_dimension" -> nodeFeatureSize,
    "hidden_size" -> hiddenSize,
    "temperature" -> temperature,
    "gradient_clip" -> gradientClip,
    "optimizer_step" -> optimizerStep,
    "tree_encoder" -> ujson.Obj(
      "leaf_encoder" -> ujson.Obj(
        "weights" -> json2d(leafEncoder),
        "bias" -> json1d(leafEncoderBias)
      ),
      "node_encoder" -> ujson.Obj(
        "weights" -> json2d(nodeEncoder),
        "bias" -> json1d(nodeEncoderBias)
      ),
      "topology_query_encoder" -> ujson.Obj(
        "weights" -> json2d(topologyQueryEncoder),
        "bias" -> json1d(topologyQueryEncoderBias)
      )
    ),
    "action_encoder" -> ujson.Obj(
      "weights" -> json2d(actionEncoder),
      "bias" -> json1d(actionEncoderBias)
    ),
    "query_encoder" -> ujson.Obj(
      "weights" -> json2d(queryEncoder),
      "bias" -> json1d(queryEncoderBias)
    ),
    "actor_head" -> ujson.Obj(
      "weights" -> json1d(actorHead)
    ),
    "value_head" -> ujson.Obj(
      "weights" -> json1d(valueHead),
      "bias" -> valueBias
    ),
    "optimizer" -> ujson.Obj(
      "leaf_encoder" -> jsonAdam2d(mLeafEncoder, vLeafEncoder),
      "leaf_encoder_bias" -> jsonAdam1d(mLeafEncoderBias, vLeafEncoderBias),
      "node_encoder" -> jsonAdam2d(mNodeEncoder, vNodeEncoder),
      "node_encoder_bias" -> jsonAdam1d(mNodeEncoderBias, vNodeEncoderBias),
      "action_encoder" -> jsonAdam2d(mActionEncoder, vActionEncoder),
      "action_encoder_bias" -> jsonAdam1d(mActionEncoderBias, vActionEncoderBias),
      "topology_query_encoder" -> jsonAdam2d(mTopologyQueryEncoder, vTopologyQueryEncoder),
      "topology_query_encoder_bias" -> jsonAdam1d(mTopologyQueryEncoderBias, vTopologyQueryEncoderBias),
      "query_encoder" -> jsonAdam2d(mQueryEncoder, vQueryEncoder),
      "query_encoder_bias" -> jsonAdam1d(mQueryEncoderBias, vQueryEncoderBias),
      "actor_head" -> jsonAdam1d(mActorHead, vActorHead),
      "value_head" -> jsonAdam1d(mValueHead, vValueHead),
      "value_bias" -> ujson.Obj(
        "m" -> mValueBias,
        "v" -> vValueBias
      )
    )
  )

  private[PrefixDeepRL] def loadFromJson(json: ujson.Value): Unit = {
    val treeEncoder = json("tree_encoder")
    loadJson2d(leafEncoder, treeEncoder("leaf_encoder")("weights"))
    loadJson1d(leafEncoderBias, treeEncoder("leaf_encoder")("bias"))
    loadJson2d(nodeEncoder, treeEncoder("node_encoder")("weights"))
    loadJson1d(nodeEncoderBias, treeEncoder("node_encoder")("bias"))
    loadJson2d(topologyQueryEncoder, treeEncoder("topology_query_encoder")("weights"))
    loadJson1d(topologyQueryEncoderBias, treeEncoder("topology_query_encoder")("bias"))

    loadJson2d(actionEncoder, json("action_encoder")("weights"))
    loadJson1d(actionEncoderBias, json("action_encoder")("bias"))
    loadJson2d(queryEncoder, json("query_encoder")("weights"))
    loadJson1d(queryEncoderBias, json("query_encoder")("bias"))
    loadJson1d(actorHead, json("actor_head")("weights"))
    loadJson1d(valueHead, json("value_head")("weights"))
    valueBias = JsonSupport.readDouble(json("value_head")("bias"))

    optimizerStep = json.obj.get("optimizer_step").map(JsonSupport.readLong).getOrElse(0L)
    json.obj.get("optimizer") match {
      case Some(optimizer) =>
        loadAdamState2d(mLeafEncoder, vLeafEncoder, optimizer("leaf_encoder"))
        loadAdamState1d(mLeafEncoderBias, vLeafEncoderBias, optimizer("leaf_encoder_bias"))
        loadAdamState2d(mNodeEncoder, vNodeEncoder, optimizer("node_encoder"))
        loadAdamState1d(mNodeEncoderBias, vNodeEncoderBias, optimizer("node_encoder_bias"))
        loadAdamState2d(mActionEncoder, vActionEncoder, optimizer("action_encoder"))
        loadAdamState1d(mActionEncoderBias, vActionEncoderBias, optimizer("action_encoder_bias"))
        loadAdamState2d(mTopologyQueryEncoder, vTopologyQueryEncoder, optimizer("topology_query_encoder"))
        loadAdamState1d(
          mTopologyQueryEncoderBias,
          vTopologyQueryEncoderBias,
          optimizer("topology_query_encoder_bias")
        )
        loadAdamState2d(mQueryEncoder, vQueryEncoder, optimizer("query_encoder"))
        loadAdamState1d(mQueryEncoderBias, vQueryEncoderBias, optimizer("query_encoder_bias"))
        loadAdamState1d(mActorHead, vActorHead, optimizer("actor_head"))
        loadAdamState1d(mValueHead, vValueHead, optimizer("value_head"))
        val valueBiasState = optimizer("value_bias")
        mValueBias = JsonSupport.readDouble(valueBiasState("m"))
        vValueBias = JsonSupport.readDouble(valueBiasState("v"))
      case None =>
        zeroOptimizerState()
    }
  }

  private def initializeParameters(): Unit = {
    val leafLimit = math.sqrt(6.0 / (leafInputSize + hiddenSize).toDouble)
    val nodeLimit = math.sqrt(6.0 / (nodeInputSize + hiddenSize).toDouble)
    val actionLimit = math.sqrt(6.0 / (actionInputSize + hiddenSize).toDouble)
    val topologyLimit = math.sqrt(6.0 / (stateInputSize + hiddenSize).toDouble)
    val queryLimit = math.sqrt(6.0 / (queryInputSize + hiddenSize).toDouble)
    val headLimit = math.sqrt(6.0 / (hiddenSize + 1).toDouble)

    initialize2d(leafEncoder, -leafLimit, leafLimit)
    initialize2d(nodeEncoder, -nodeLimit, nodeLimit)
    initialize2d(actionEncoder, -actionLimit, actionLimit)
    initialize2d(topologyQueryEncoder, -topologyLimit, topologyLimit)
    initialize2d(queryEncoder, -queryLimit, queryLimit)
    initialize1d(leafEncoderBias, 0.0)
    initialize1d(nodeEncoderBias, 0.0)
    initialize1d(actionEncoderBias, 0.0)
    initialize1d(topologyQueryEncoderBias, 0.0)
    initialize1d(queryEncoderBias, 0.0)

    var idx = 0
    while (idx < hiddenSize) {
      actorHead(idx) = uniform(-headLimit, headLimit)
      valueHead(idx) = uniform(-headLimit, headLimit)
      idx += 1
    }
    valueBias = 0.0
    zeroOptimizerState()
  }

  private def zeroOptimizerState(): Unit = {
    initialize2d(mLeafEncoder, 0.0, 0.0)
    initialize2d(vLeafEncoder, 0.0, 0.0)
    initialize1d(mLeafEncoderBias, 0.0)
    initialize1d(vLeafEncoderBias, 0.0)
    initialize2d(mNodeEncoder, 0.0, 0.0)
    initialize2d(vNodeEncoder, 0.0, 0.0)
    initialize1d(mNodeEncoderBias, 0.0)
    initialize1d(vNodeEncoderBias, 0.0)
    initialize2d(mActionEncoder, 0.0, 0.0)
    initialize2d(vActionEncoder, 0.0, 0.0)
    initialize1d(mActionEncoderBias, 0.0)
    initialize1d(vActionEncoderBias, 0.0)
    initialize2d(mTopologyQueryEncoder, 0.0, 0.0)
    initialize2d(vTopologyQueryEncoder, 0.0, 0.0)
    initialize1d(mTopologyQueryEncoderBias, 0.0)
    initialize1d(vTopologyQueryEncoderBias, 0.0)
    initialize2d(mQueryEncoder, 0.0, 0.0)
    initialize2d(vQueryEncoder, 0.0, 0.0)
    initialize1d(mQueryEncoderBias, 0.0)
    initialize1d(vQueryEncoderBias, 0.0)
    initialize1d(mActorHead, 0.0)
    initialize1d(vActorHead, 0.0)
    initialize1d(mValueHead, 0.0)
    initialize1d(vValueHead, 0.0)
    mValueBias = 0.0
    vValueBias = 0.0
    optimizerStep = 0L
  }

  private def encodeExistingOutputs(
    width:           Int,
    existingOutputs: IndexedSeq[PrefixTree]
  ): EncodedTopology = {
    val caches = mutable.LinkedHashMap.empty[String, TreeEmbeddingCache]
    val dagUseCounts = mutable.LinkedHashMap.empty[String, Int]
    val outputSignatures = existingOutputs.map { tree =>
      collectInternalNodeUses(tree, dagUseCounts)
      encodeTree(width, tree, caches).signature
    }.toVector
    val dagSignatures = dagUseCounts.keys.toVector
    val dagWeights = if (dagSignatures.isEmpty) {
      Array.empty[Double]
    } else {
      val totalUses = dagUseCounts.values.sum.toDouble
      dagSignatures.map(signature => dagUseCounts(signature).toDouble / totalUses).toArray
    }

    EncodedTopology(
      outputSignatures = outputSignatures,
      dagSignatures = dagSignatures,
      dagWeights = dagWeights,
      treeCaches = caches.toMap,
      treeCacheOrder = caches.keys.toVector
    )
  }

  private def encodeTree(
    width:  Int,
    tree:   PrefixTree,
    caches: mutable.LinkedHashMap[String, TreeEmbeddingCache]
  ): TreeEmbeddingCache = {
    val signature = tree.signature
    caches.getOrElseUpdate(
      signature, {
        tree match {
          case Leaf(index) =>
            val input = LeafFeatureEncoder.encode(width, index)
            val embedding = tanhLayer(leafEncoder, leafEncoderBias, input)
            TreeEmbeddingCache(
              signature = signature,
              input = input,
              embedding = embedding,
              isLeaf = true,
              leftSignature = None,
              rightSignature = None
            )

          case node @ Node(left, right) =>
            val leftCache = encodeTree(width, left, caches)
            val rightCache = encodeTree(width, right, caches)
            val input = concat(leftCache.embedding, rightCache.embedding, NodeFeatureEncoder.encode(width, node))
            val embedding = tanhLayer(nodeEncoder, nodeEncoderBias, input)
            TreeEmbeddingCache(
              signature = signature,
              input = input,
              embedding = embedding,
              isLeaf = false,
              leftSignature = Some(leftCache.signature),
              rightSignature = Some(rightCache.signature)
            )
        }
      }
    )
  }

  private def collectInternalNodeUses(
    tree:         PrefixTree,
    dagUseCounts: mutable.LinkedHashMap[String, Int]
  ): Unit = tree match {
    case Leaf(_) =>
    case node @ Node(left, right) =>
      dagUseCounts.update(node.signature, dagUseCounts.getOrElse(node.signature, 0) + 1)
      collectInternalNodeUses(left, dagUseCounts)
      collectInternalNodeUses(right, dagUseCounts)
  }

  private def computeActionAttentionWeights(actionEmbeddings: IndexedSeq[Array[Double]]): Array[Array[Double]] = {
    require(actionEmbeddings.nonEmpty, "cannot compute action attention over an empty set")

    val weights = Array.ofDim[Double](actionEmbeddings.length, actionEmbeddings.length)
    var row = 0
    while (row < actionEmbeddings.length) {
      val logits = Array.ofDim[Double](actionEmbeddings.length)
      var col = 0
      while (col < actionEmbeddings.length) {
        logits(col) = interactionScale * dot(actionEmbeddings(row), actionEmbeddings(col))
        col += 1
      }
      weights(row) = softmax(logits, temperature = 1.0)
      row += 1
    }
    weights
  }

  private def backprop(
    step:          SampledDecision,
    advantage:     Double,
    valueResidual: Double,
    grads:         NetworkGradients
  ): Unit = {
    val candidateCount = step.evaluations.length
    val query = step.cache.query
    val queryInput = step.cache.queryInput
    val queryInputGrad = Array.fill(queryInputSize)(0.0)
    val queryGrad = Array.fill(hiddenSize)(0.0)
    val contextualEmbeddingGrads = Array.fill(candidateCount, hiddenSize)(0.0)

    var idx = 0
    while (idx < candidateCount) {
      val evaluation = step.evaluations(idx)
      val indicator = if (idx == step.actionIndex) 1.0 else 0.0
      val logitGrad = advantage * (step.probabilities(idx) - indicator) / temperature

      var h = 0
      while (h < hiddenSize) {
        grads.actorHead(h) += logitGrad * evaluation.contextualEmbedding(h)
        queryGrad(h) += logitGrad * interactionScale * evaluation.contextualEmbedding(h)
        contextualEmbeddingGrads(idx)(h) += logitGrad * (actorHead(h) + interactionScale * query(h))
        h += 1
      }
      idx += 1
    }

    grads.valueBias += valueResidual
    var h = 0
    while (h < hiddenSize) {
      grads.valueHead(h) += valueResidual * query(h)
      queryGrad(h) += valueResidual * valueHead(h)
      h += 1
    }

    h = 0
    while (h < hiddenSize) {
      val dPre = queryGrad(h) * (1.0 - query(h) * query(h))
      grads.queryEncoderBias(h) += dPre

      var k = 0
      while (k < queryInputSize) {
        grads.queryEncoder(h)(k) += dPre * queryInput(k)
        queryInputGrad(k) += dPre * queryEncoder(h)(k)
        k += 1
      }
      h += 1
    }

    val topologySummaryGrad = Array.fill(hiddenSize)(0.0)
    val dagSummaryGrad = Array.fill(hiddenSize)(0.0)
    val actionSummaryGrad = Array.fill(hiddenSize)(0.0)
    h = 0
    while (h < hiddenSize) {
      topologySummaryGrad(h) = queryInputGrad(stateInputSize + h)
      dagSummaryGrad(h) = queryInputGrad(stateInputSize + hiddenSize + h)
      actionSummaryGrad(h) = queryInputGrad(stateInputSize + 2 * hiddenSize + h)
      h += 1
    }

    h = 0
    while (h < hiddenSize) {
      val meanGrad = actionSummaryGrad(h) / candidateCount.toDouble
      idx = 0
      while (idx < candidateCount) {
        contextualEmbeddingGrads(idx)(h) += meanGrad
        idx += 1
      }
      h += 1
    }

    backpropActionContext(step, contextualEmbeddingGrads, grads)
    backpropTopology(step.cache, topologySummaryGrad, dagSummaryGrad, grads)
  }

  private def backpropActionContext(
    step:                     SampledDecision,
    contextualEmbeddingGrads: Array[Array[Double]],
    grads:                    NetworkGradients
  ): Unit = {
    val candidateCount = step.evaluations.length
    val baseEmbeddings = step.evaluations.map(_.actionEmbedding)
    val contextualEmbeddings = step.evaluations.map(_.contextualEmbedding)
    val baseEmbeddingGrads = Array.fill(candidateCount, hiddenSize)(0.0)

    if (step.cache.actionContextMode == NeuralArchitectureConfig.DefaultActionContextMode) {
      val attentionWeights = step.cache.actionAttentionWeights
      var row = 0
      while (row < candidateCount) {
        val contextualGrad = Array.fill(hiddenSize)(0.0)
        var h = 0
        while (h < hiddenSize) {
          val grad =
            contextualEmbeddingGrads(row)(h) * (1.0 - contextualEmbeddings(row)(h) * contextualEmbeddings(row)(h))
          contextualGrad(h) = grad
          baseEmbeddingGrads(row)(h) += grad
          h += 1
        }

        val rowWeights = attentionWeights(row)
        val weightGrads = Array.fill(candidateCount)(0.0)
        var col = 0
        while (col < candidateCount) {
          h = 0
          while (h < hiddenSize) {
            baseEmbeddingGrads(col)(h) += contextualGrad(h) * rowWeights(col)
            h += 1
          }
          weightGrads(col) = dot(contextualGrad, baseEmbeddings(col))
          col += 1
        }

        var meanWeightGrad = 0.0
        col = 0
        while (col < candidateCount) {
          meanWeightGrad += rowWeights(col) * weightGrads(col)
          col += 1
        }

        col = 0
        while (col < candidateCount) {
          val scoreGrad = rowWeights(col) * (weightGrads(col) - meanWeightGrad)
          val scaledScoreGrad = interactionScale * scoreGrad
          h = 0
          while (h < hiddenSize) {
            baseEmbeddingGrads(row)(h) += scaledScoreGrad * baseEmbeddings(col)(h)
            baseEmbeddingGrads(col)(h) += scaledScoreGrad * baseEmbeddings(row)(h)
            h += 1
          }
          col += 1
        }

        row += 1
      }
    } else {
      val meanEmbeddingGrad = Array.fill(hiddenSize)(0.0)
      var row = 0
      while (row < candidateCount) {
        var h = 0
        while (h < hiddenSize) {
          val grad =
            contextualEmbeddingGrads(row)(h) * (1.0 - contextualEmbeddings(row)(h) * contextualEmbeddings(row)(h))
          baseEmbeddingGrads(row)(h) += grad
          meanEmbeddingGrad(h) += grad
          h += 1
        }
        row += 1
      }

      var h = 0
      while (h < hiddenSize) {
        val sharedGrad = meanEmbeddingGrad(h) / candidateCount.toDouble
        var row = 0
        while (row < candidateCount) {
          baseEmbeddingGrads(row)(h) += sharedGrad
          row += 1
        }
        h += 1
      }
    }

    var row = 0
    while (row < candidateCount) {
      val evaluation = step.evaluations(row)
      var h = 0
      while (h < hiddenSize) {
        val dPre = baseEmbeddingGrads(row)(h) * (1.0 - evaluation.actionEmbedding(h) * evaluation.actionEmbedding(h))
        grads.actionEncoderBias(h) += dPre

        var k = 0
        while (k < actionInputSize) {
          grads.actionEncoder(h)(k) += dPre * evaluation.actionInput(k)
          k += 1
        }
        h += 1
      }
      row += 1
    }
  }

  private def backpropTopology(
    cache:               DecisionForwardCache,
    topologySummaryGrad: Array[Double],
    dagSummaryGrad:      Array[Double],
    grads:               NetworkGradients
  ): Unit = {
    val outputCount = cache.outputSignatures.length
    val outputEmbeddingGrads = mutable.HashMap.empty[String, Array[Double]]
    val topologyQueryGrad = Array.fill(hiddenSize)(0.0)
    val attentionGrads = Array.fill(outputCount)(0.0)

    var idx = 0
    while (idx < outputCount) {
      val signature = cache.outputSignatures(idx)
      val embedding = cache.treeCaches(signature).embedding
      val grad = outputEmbeddingGrads.getOrElseUpdate(signature, Array.fill(hiddenSize)(0.0))

      var h = 0
      while (h < hiddenSize) {
        grad(h) += topologySummaryGrad(h) * cache.outputAttentionWeights(idx)
        h += 1
      }
      attentionGrads(idx) = dot(topologySummaryGrad, embedding)
      idx += 1
    }

    idx = 0
    while (idx < cache.dagSignatures.length) {
      val signature = cache.dagSignatures(idx)
      val grad = outputEmbeddingGrads.getOrElseUpdate(signature, Array.fill(hiddenSize)(0.0))
      val weight = cache.dagSummaryWeights(idx)
      var h = 0
      while (h < hiddenSize) {
        grad(h) += dagSummaryGrad(h) * weight
        h += 1
      }
      idx += 1
    }

    var meanAttentionGrad = 0.0
    idx = 0
    while (idx < outputCount) {
      meanAttentionGrad += cache.outputAttentionWeights(idx) * attentionGrads(idx)
      idx += 1
    }

    idx = 0
    while (idx < outputCount) {
      val signature = cache.outputSignatures(idx)
      val embedding = cache.treeCaches(signature).embedding
      val grad = outputEmbeddingGrads(signature)
      val scoreGrad = cache.outputAttentionWeights(idx) * (attentionGrads(idx) - meanAttentionGrad)

      var h = 0
      while (h < hiddenSize) {
        topologyQueryGrad(h) += interactionScale * scoreGrad * embedding(h)
        grad(h) += interactionScale * scoreGrad * cache.topologyQuery(h)
        h += 1
      }
      idx += 1
    }

    if (cache.dagSummaryMode == "attention-weighted") {
      val dagAttentionGrads = Array.fill(cache.dagSignatures.length)(0.0)
      idx = 0
      while (idx < cache.dagSignatures.length) {
        val signature = cache.dagSignatures(idx)
        val embedding = cache.treeCaches(signature).embedding
        dagAttentionGrads(idx) = dot(dagSummaryGrad, embedding)
        idx += 1
      }

      var meanDagAttentionGrad = 0.0
      idx = 0
      while (idx < cache.dagSignatures.length) {
        meanDagAttentionGrad += cache.dagSummaryWeights(idx) * dagAttentionGrads(idx)
        idx += 1
      }

      idx = 0
      while (idx < cache.dagSignatures.length) {
        val signature = cache.dagSignatures(idx)
        val embedding = cache.treeCaches(signature).embedding
        val grad = outputEmbeddingGrads(signature)
        val scoreGrad = cache.dagSummaryWeights(idx) * (dagAttentionGrads(idx) - meanDagAttentionGrad)

        var h = 0
        while (h < hiddenSize) {
          topologyQueryGrad(h) += interactionScale * scoreGrad * embedding(h)
          grad(h) += interactionScale * scoreGrad * cache.topologyQuery(h)
          h += 1
        }
        idx += 1
      }
    }

    var h = 0
    while (h < hiddenSize) {
      val dPre = topologyQueryGrad(h) * (1.0 - cache.topologyQuery(h) * cache.topologyQuery(h))
      grads.topologyQueryEncoderBias(h) += dPre

      var k = 0
      while (k < stateInputSize) {
        grads.topologyQueryEncoder(h)(k) += dPre * cache.stateInput(k)
        k += 1
      }
      h += 1
    }

    backpropTreeEmbeddings(cache, outputEmbeddingGrads.toMap, grads)
  }

  private def backpropTreeEmbeddings(
    cache:              DecisionForwardCache,
    rootEmbeddingGrads: Map[String, Array[Double]],
    grads:              NetworkGradients
  ): Unit = {
    val embeddingGrads = mutable.HashMap.empty[String, Array[Double]]
    cache.treeCacheOrder.foreach { signature =>
      embeddingGrads.update(signature, Array.fill(hiddenSize)(0.0))
    }

    rootEmbeddingGrads.foreach { case (signature, grad) =>
      val target = embeddingGrads.getOrElseUpdate(signature, Array.fill(hiddenSize)(0.0))
      addInPlace(target, grad)
    }

    cache.treeCacheOrder.reverseIterator.foreach { signature =>
      val treeCache = cache.treeCaches(signature)
      val embeddingGrad = embeddingGrads(signature)

      if (treeCache.isLeaf) {
        var h = 0
        while (h < hiddenSize) {
          val dPre = embeddingGrad(h) * (1.0 - treeCache.embedding(h) * treeCache.embedding(h))
          grads.leafEncoderBias(h) += dPre

          var k = 0
          while (k < leafInputSize) {
            grads.leafEncoder(h)(k) += dPre * treeCache.input(k)
            k += 1
          }
          h += 1
        }
      } else {
        val inputGrad = Array.fill(nodeInputSize)(0.0)
        var h = 0
        while (h < hiddenSize) {
          val dPre = embeddingGrad(h) * (1.0 - treeCache.embedding(h) * treeCache.embedding(h))
          grads.nodeEncoderBias(h) += dPre

          var k = 0
          while (k < nodeInputSize) {
            grads.nodeEncoder(h)(k) += dPre * treeCache.input(k)
            inputGrad(k) += dPre * nodeEncoder(h)(k)
            k += 1
          }
          h += 1
        }

        val leftGrad = embeddingGrads(treeCache.leftSignature.get)
        val rightGrad = embeddingGrads(treeCache.rightSignature.get)
        h = 0
        while (h < hiddenSize) {
          leftGrad(h) += inputGrad(h)
          rightGrad(h) += inputGrad(hiddenSize + h)
          h += 1
        }
      }
    }
  }

  private def tanhLayer(
    weights: Array[Array[Double]],
    bias:    Array[Double],
    input:   Array[Double]
  ): Array[Double] = {
    val output = Array.fill(weights.length)(0.0)

    var row = 0
    while (row < weights.length) {
      var sum = bias(row)
      var col = 0
      while (col < input.length) {
        sum += weights(row)(col) * input(col)
        col += 1
      }
      output(row) = math.tanh(sum)
      row += 1
    }

    output
  }

  private def tanhVector(values: Array[Double]): Array[Double] = {
    val out = Array.ofDim[Double](values.length)
    var idx = 0
    while (idx < values.length) {
      out(idx) = math.tanh(values(idx))
      idx += 1
    }
    out
  }

  private def softmax(logits: Array[Double], temperature: Double): Array[Double] = {
    val scaled = logits.map(_ / temperature)
    val max = scaled.max
    val exp = scaled.map(v => math.exp(v - max))
    val sum = exp.sum
    if (sum == 0.0 || !sum.isFinite) {
      Array.fill(logits.length)(1.0 / logits.length.toDouble)
    } else {
      exp.map(_ / sum)
    }
  }

  private def averageEmbeddings(embeddings: IndexedSeq[Array[Double]]): Array[Double] = {
    require(embeddings.nonEmpty, "cannot average an empty embedding set")
    val out = Array.fill(hiddenSize)(0.0)

    var idx = 0
    while (idx < embeddings.length) {
      addInPlace(out, embeddings(idx))
      idx += 1
    }

    idx = 0
    while (idx < hiddenSize) {
      out(idx) /= embeddings.length.toDouble
      idx += 1
    }
    out
  }

  private def weightedAverage(embeddings: IndexedSeq[Array[Double]], weights: Array[Double]): Array[Double] = {
    require(embeddings.nonEmpty, "cannot average an empty embedding set")
    require(embeddings.length == weights.length, s"weight count mismatch: ${weights.length} != ${embeddings.length}")

    val out = Array.fill(hiddenSize)(0.0)
    var idx = 0
    while (idx < embeddings.length) {
      addScaledInPlace(out, embeddings(idx), weights(idx))
      idx += 1
    }
    out
  }

  private def add(left: Array[Double], right: Array[Double]): Array[Double] = {
    require(left.length == right.length, s"vector length mismatch: ${left.length} != ${right.length}")
    val out = Array.ofDim[Double](left.length)
    var idx = 0
    while (idx < left.length) {
      out(idx) = left(idx) + right(idx)
      idx += 1
    }
    out
  }

  private def concat(left: Array[Double], right: Array[Double], third: Array[Double]): Array[Double] = {
    val out = Array.ofDim[Double](left.length + right.length + third.length)
    System.arraycopy(left, 0, out, 0, left.length)
    System.arraycopy(right, 0, out, left.length, right.length)
    System.arraycopy(third, 0, out, left.length + right.length, third.length)
    out
  }

  private def concat(
    left:   Array[Double],
    right:  Array[Double],
    third:  Array[Double],
    fourth: Array[Double]
  ): Array[Double] = {
    val out = Array.ofDim[Double](left.length + right.length + third.length + fourth.length)
    System.arraycopy(left, 0, out, 0, left.length)
    System.arraycopy(right, 0, out, left.length, right.length)
    System.arraycopy(third, 0, out, left.length + right.length, third.length)
    System.arraycopy(fourth, 0, out, left.length + right.length + third.length, fourth.length)
    out
  }

  private def dot(left: Array[Double], right: Array[Double]): Double = {
    var sum = 0.0
    var idx = 0
    while (idx < left.length) {
      sum += left(idx) * right(idx)
      idx += 1
    }
    sum
  }

  private def addInPlace(target: Array[Double], source: Array[Double]): Unit = {
    var idx = 0
    while (idx < target.length) {
      target(idx) += source(idx)
      idx += 1
    }
  }

  private def addScaledInPlace(target: Array[Double], source: Array[Double], scale: Double): Unit = {
    var idx = 0
    while (idx < target.length) {
      target(idx) += scale * source(idx)
      idx += 1
    }
  }

  private def export1d(values: Array[Double], out: Array[Double], start: Int): Int = {
    var idx = start
    var i = 0
    while (i < values.length) {
      out(idx) = values(i)
      idx += 1
      i += 1
    }
    idx
  }

  private def export2d(values: Array[Array[Double]], out: Array[Double], start: Int): Int = {
    var idx = start
    var row = 0
    while (row < values.length) {
      idx = export1d(values(row), out, idx)
      row += 1
    }
    idx
  }

  private def import1d(source: Array[Double], start: Int, out: Array[Double]): Int = {
    var idx = start
    var i = 0
    while (i < out.length) {
      out(i) = source(idx)
      idx += 1
      i += 1
    }
    idx
  }

  private def import2d(source: Array[Double], start: Int, out: Array[Array[Double]]): Int = {
    var idx = start
    var row = 0
    while (row < out.length) {
      idx = import1d(source, idx, out(row))
      row += 1
    }
    idx
  }

  private def flattenGradients(grads: NetworkGradients): Array[Double] = {
    val out = Array.ofDim[Double](parameterCount)
    var idx = 0

    idx = export2d(grads.leafEncoder, out, idx)
    idx = export1d(grads.leafEncoderBias, out, idx)
    idx = export2d(grads.nodeEncoder, out, idx)
    idx = export1d(grads.nodeEncoderBias, out, idx)
    idx = export2d(grads.actionEncoder, out, idx)
    idx = export1d(grads.actionEncoderBias, out, idx)
    idx = export2d(grads.topologyQueryEncoder, out, idx)
    idx = export1d(grads.topologyQueryEncoderBias, out, idx)
    idx = export2d(grads.queryEncoder, out, idx)
    idx = export1d(grads.queryEncoderBias, out, idx)
    idx = export1d(grads.actorHead, out, idx)
    idx = export1d(grads.valueHead, out, idx)
    out(idx) = grads.valueBias
    out
  }

  private def applyGradients(grads: NetworkGradients, learningRate: Double): Unit = {
    optimizerStep += 1L
    val step = optimizerStep.toDouble

    applyAdam2d(leafEncoder, mLeafEncoder, vLeafEncoder, grads.leafEncoder, step, learningRate)
    applyAdam1d(leafEncoderBias, mLeafEncoderBias, vLeafEncoderBias, grads.leafEncoderBias, step, learningRate)
    applyAdam2d(nodeEncoder, mNodeEncoder, vNodeEncoder, grads.nodeEncoder, step, learningRate)
    applyAdam1d(nodeEncoderBias, mNodeEncoderBias, vNodeEncoderBias, grads.nodeEncoderBias, step, learningRate)
    applyAdam2d(actionEncoder, mActionEncoder, vActionEncoder, grads.actionEncoder, step, learningRate)
    applyAdam1d(actionEncoderBias, mActionEncoderBias, vActionEncoderBias, grads.actionEncoderBias, step, learningRate)
    applyAdam2d(
      topologyQueryEncoder,
      mTopologyQueryEncoder,
      vTopologyQueryEncoder,
      grads.topologyQueryEncoder,
      step,
      learningRate
    )
    applyAdam1d(
      topologyQueryEncoderBias,
      mTopologyQueryEncoderBias,
      vTopologyQueryEncoderBias,
      grads.topologyQueryEncoderBias,
      step,
      learningRate
    )
    applyAdam2d(queryEncoder, mQueryEncoder, vQueryEncoder, grads.queryEncoder, step, learningRate)
    applyAdam1d(queryEncoderBias, mQueryEncoderBias, vQueryEncoderBias, grads.queryEncoderBias, step, learningRate)
    applyAdam1d(actorHead, mActorHead, vActorHead, grads.actorHead, step, learningRate)
    applyAdam1d(valueHead, mValueHead, vValueHead, grads.valueHead, step, learningRate)

    val (mValueBiasNew, vValueBiasNew, dValueBias) = adamUpdate(
      grad = grads.valueBias,
      mPrev = mValueBias,
      vPrev = vValueBias,
      t = step,
      learningRate = learningRate
    )
    mValueBias = mValueBiasNew
    vValueBias = vValueBiasNew
    valueBias -= dValueBias
  }

  private def applyAdam1d(
    params:       Array[Double],
    mState:       Array[Double],
    vState:       Array[Double],
    grads:        Array[Double],
    step:         Double,
    learningRate: Double
  ): Unit = {
    var idx = 0
    while (idx < params.length) {
      val (m, v, delta) = adamUpdate(
        grad = grads(idx),
        mPrev = mState(idx),
        vPrev = vState(idx),
        t = step,
        learningRate = learningRate
      )
      mState(idx) = m
      vState(idx) = v
      params(idx) -= delta
      idx += 1
    }
  }

  private def applyAdam2d(
    params:       Array[Array[Double]],
    mState:       Array[Array[Double]],
    vState:       Array[Array[Double]],
    grads:        Array[Array[Double]],
    step:         Double,
    learningRate: Double
  ): Unit = {
    var row = 0
    while (row < params.length) {
      applyAdam1d(params(row), mState(row), vState(row), grads(row), step, learningRate)
      row += 1
    }
  }

  private def adamUpdate(
    grad:         Double,
    mPrev:        Double,
    vPrev:        Double,
    t:            Double,
    learningRate: Double
  ): (Double, Double, Double) = {
    val beta1 = 0.9
    val beta2 = 0.999
    val epsilon = 1e-8

    val m = beta1 * mPrev + (1.0 - beta1) * grad
    val v = beta2 * vPrev + (1.0 - beta2) * grad * grad
    val mHat = m / (1.0 - math.pow(beta1, t))
    val vHat = v / (1.0 - math.pow(beta2, t))
    val delta = learningRate * mHat / (math.sqrt(vHat) + epsilon)
    (m, v, delta)
  }

  private def initialize1d(values: Array[Double], constant: Double): Unit = {
    var idx = 0
    while (idx < values.length) {
      values(idx) = constant
      idx += 1
    }
  }

  private def initialize2d(values: Array[Array[Double]], low: Double, high: Double): Unit = {
    var row = 0
    while (row < values.length) {
      var col = 0
      while (col < values(row).length) {
        values(row)(col) = uniform(low, high)
        col += 1
      }
      row += 1
    }
  }

  private def loadJson1d(target: Array[Double], value: ujson.Value): Unit = {
    assign1d(target, readJson1d(value, target.length))
  }

  private def loadJson2d(target: Array[Array[Double]], value: ujson.Value): Unit = {
    assign2d(target, readJson2d(value, target.length, if (target.isEmpty) 0 else target.head.length))
  }

  private def loadAdamState1d(mTarget: Array[Double], vTarget: Array[Double], value: ujson.Value): Unit = {
    loadJson1d(mTarget, value("m"))
    loadJson1d(vTarget, value("v"))
  }

  private def loadAdamState2d(
    mTarget: Array[Array[Double]],
    vTarget: Array[Array[Double]],
    value:   ujson.Value
  ): Unit = {
    loadJson2d(mTarget, value("m"))
    loadJson2d(vTarget, value("v"))
  }

  private def assign1d(target: Array[Double], source: Array[Double]): Unit = {
    require(target.length == source.length, s"vector length mismatch: ${target.length} != ${source.length}")
    var idx = 0
    while (idx < target.length) {
      target(idx) = source(idx)
      idx += 1
    }
  }

  private def assign2d(target: Array[Array[Double]], source: Array[Array[Double]]): Unit = {
    require(target.length == source.length, s"row count mismatch: ${target.length} != ${source.length}")
    var row = 0
    while (row < target.length) {
      assign1d(target(row), source(row))
      row += 1
    }
  }

  private def readJson1d(value: ujson.Value, expectedLength: Int): Array[Double] = value match {
    case ujson.Arr(values) =>
      require(values.length == expectedLength, s"Expected length ${expectedLength}, found ${values.length}")
      values.iterator.map(JsonSupport.readDouble).toArray
    case other =>
      throw new IllegalArgumentException(s"Expected JSON array, found ${other}")
  }

  private def readJson2d(value: ujson.Value, expectedRows: Int, expectedCols: Int): Array[Array[Double]] = value match {
    case ujson.Arr(rows) =>
      require(rows.length == expectedRows, s"Expected ${expectedRows} rows, found ${rows.length}")
      rows.iterator.map(row => readJson1d(row, expectedCols)).toArray
    case other =>
      throw new IllegalArgumentException(s"Expected JSON 2D array, found ${other}")
  }

  private def json1d(values: Array[Double]): ujson.Value = ujson.Arr.from(values.toSeq.map(ujson.Num(_)))

  private def json2d(values: Array[Array[Double]]): ujson.Value =
    ujson.Arr.from(values.toSeq.map(row => ujson.Arr.from(row.toSeq.map(ujson.Num(_)))))

  private def jsonAdam1d(mState: Array[Double], vState: Array[Double]): ujson.Value = ujson.Obj(
    "m" -> json1d(mState),
    "v" -> json1d(vState)
  )

  private def jsonAdam2d(mState: Array[Array[Double]], vState: Array[Array[Double]]): ujson.Value = ujson.Obj(
    "m" -> json2d(mState),
    "v" -> json2d(vState)
  )

  private def uniform(low: Double, high: Double): Double = low + random.nextDouble() * (high - low)
}
