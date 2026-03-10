package PrefixRLCore

import PrefixAdderLib.{DependentTopology, Leaf, Node, PrefixTree}

import scala.collection.mutable

final case class DecisionContext(
  outputIndex: Int,
  segmentLow:  Int,
  segmentHigh: Int,
  root:        Boolean
) {
  require(segmentLow < segmentHigh, s"decision context requires at least one split: ${this}")

  val actionCount: Int = segmentHigh - segmentLow

  def absoluteSplit(actionIndex: Int): Int = segmentLow + actionIndex

  def key: String = s"${outputIndex}:${segmentLow}-${segmentHigh}:${if (root) "root" else "suffix"}"
}

trait SplitDecision {
  def context:     DecisionContext
  def actionIndex: Int

  final def absoluteSplit: Int = context.absoluteSplit(actionIndex)
}

trait SplitPolicy[D <: SplitDecision] {
  def sample(
    searchState:     SearchPolicyState,
    width:           Int,
    context:         DecisionContext,
    existingOutputs: IndexedSeq[PrefixTree]
  ): D
}

final case class EpisodeSample[D <: SplitDecision](
  topology:  DependentTopology,
  decisions: Vector[D]
)

final class DependentTreeEpisodeBuilder[D <: SplitDecision](policy: SplitPolicy[D]) {
  private def validatedSplit(context: DecisionContext, choice: D): Int = {
    require(
      choice.context == context,
      s"policy returned decision for ${choice.context}, expected ${context}"
    )
    require(
      choice.actionIndex >= 0 && choice.actionIndex < context.actionCount,
      s"invalid action index ${choice.actionIndex} for ${context}; expected 0..${context.actionCount - 1}"
    )

    val split = choice.absoluteSplit
    require(
      split >= context.segmentLow && split < context.segmentHigh,
      s"invalid split ${split} for ${context}; expected ${context.segmentLow}..${context.segmentHigh - 1}"
    )
    split
  }

  def sample(width: Int, searchState: SearchPolicyState = SearchPolicyState.empty): EpisodeSample[D] = {
    require(width >= 1, s"width must be >= 1, got ${width}")

    val outputs = mutable.ArrayBuffer.empty[PrefixTree]
    val decisions = mutable.ArrayBuffer.empty[D]

    outputs += Leaf(0)

    for (outputIndex <- 1 until width) {
      val rootContext = DecisionContext(
        outputIndex = outputIndex,
        segmentLow = 0,
        segmentHigh = outputIndex,
        root = true
      )
      val rootChoice = policy.sample(searchState, width, rootContext, outputs.toVector)
      decisions += rootChoice

      val split = validatedSplit(rootContext, rootChoice)
      val left = outputs(split)
      val right = if (split + 1 == outputIndex) {
        Leaf(outputIndex)
      } else {
        buildSuffix(searchState, width, outputIndex, split + 1, outputIndex, outputs.toVector, decisions)
      }
      outputs += Node(left, right)
    }

    EpisodeSample(
      topology = DependentTopology(width, outputs.toVector),
      decisions = decisions.toVector
    )
  }

  private def buildSuffix(
    searchState: SearchPolicyState,
    width:       Int,
    outputIndex: Int,
    low:         Int,
    high:        Int,
    outputs:     Vector[PrefixTree],
    decisions:   mutable.ArrayBuffer[D]
  ): PrefixTree = {
    if (low == high) {
      Leaf(low)
    } else {
      val context = DecisionContext(
        outputIndex = outputIndex,
        segmentLow = low,
        segmentHigh = high,
        root = false
      )
      val choice = policy.sample(searchState, width, context, outputs)
      decisions += choice

      val split = validatedSplit(context, choice)
      val left = buildSuffix(searchState, width, outputIndex, low, split, outputs, decisions)
      val right = buildSuffix(searchState, width, outputIndex, split + 1, high, outputs, decisions)
      Node(left, right)
    }
  }
}
