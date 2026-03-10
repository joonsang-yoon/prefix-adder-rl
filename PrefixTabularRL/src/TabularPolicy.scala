package PrefixTabularRL

import PrefixAdderLib.PrefixTree
import PrefixRLCore.{DecisionContext, SearchPolicyState, SplitDecision, SplitPolicy}
import PrefixUtils.JsonSupport

import scala.collection.mutable
import scala.util.Random

final case class TabularTrainingStats(
  advantage:      Double,
  averageEntropy: Double
) {
  def toJson: ujson.Value = ujson.Obj(
    "advantage" -> advantage,
    "average_entropy" -> averageEntropy
  )
}

final case class SearchStateBucket(
  phaseBucket:     Int,
  frontierBucket:  Int,
  duplicateBucket: Int,
  spreadBucket:    Int
) {
  def key: String =
    s"phase=${phaseBucket}:frontier=${frontierBucket}:duplicate=${duplicateBucket}:spread=${spreadBucket}"
}

object SearchStateBucket {
  def from(state: SearchPolicyState): SearchStateBucket = SearchStateBucket(
    phaseBucket = bucketizeCount(state.completedEvaluations),
    frontierBucket = bucketizeUnitInterval(state.frontierFraction),
    duplicateBucket = bucketizeUnitInterval(state.duplicateRate),
    spreadBucket = bucketizeSpread(state.frontierSpread)
  )

  private def bucketizeCount(count: Int): Int = {
    if (count <= 0) {
      0
    } else {
      math.min(5, math.floor(math.log(count.toDouble) / math.log(2.0)).toInt + 1)
    }
  }

  private def bucketizeUnitInterval(value: Double): Int = {
    val clipped = value.max(0.0).min(1.0)

    if (clipped >= 1.0) {
      4
    } else if (clipped < 0.25) {
      0
    } else if (clipped < 0.50) {
      1
    } else if (clipped < 0.75) {
      2
    } else {
      3
    }
  }

  private def bucketizeSpread(value: Double): Int = {
    if (value <= 0.0) {
      0
    } else if (value < 0.15) {
      1
    } else if (value < 0.35) {
      2
    } else if (value < 0.7) {
      3
    } else {
      4
    }
  }
}

final case class TabularContextKey(
  context:      DecisionContext,
  searchBucket: SearchStateBucket
) {
  def key: String = s"${context.key}:${searchBucket.key}"
}

final case class TabularDecision(
  tableKey:      TabularContextKey,
  context:       DecisionContext,
  actionIndex:   Int,
  probabilities: Vector[Double]
) extends SplitDecision {
  lazy val entropy: Double = probabilities.filter(_ > 0.0).map(p => -p * math.log(p)).sum
}

object TabularSoftmaxPolicy {
  def load(path: os.Path, random: Random): TabularSoftmaxPolicy =
    fromJson(ujson.read(os.read(path)), random)

  def fromJson(json: ujson.Value, random: Random): TabularSoftmaxPolicy = {
    val temperature = JsonSupport.readDouble(json("temperature"))
    val policy = new TabularSoftmaxPolicy(random, temperature)
    policy.loadFromJson(json)
    policy
  }
}

final class TabularSoftmaxPolicy(
  random:      Random,
  temperature: Double
) extends SplitPolicy[TabularDecision] {
  require(temperature > 0.0, s"temperature must be > 0, got ${temperature}")

  def temperatureValue: Double = temperature

  private val logits = mutable.HashMap.empty[TabularContextKey, Vector[Double]]

  def distribution(context: DecisionContext, searchState: SearchPolicyState): Vector[Double] = {
    val key = TabularContextKey(context, SearchStateBucket.from(searchState))
    val prefs = logits.getOrElseUpdate(key, Vector.fill(context.actionCount)(0.0))
    softmax(prefs)
  }

  def distribution(context: DecisionContext): Vector[Double] = distribution(context, SearchPolicyState.empty)

  override def sample(
    searchState:     SearchPolicyState,
    width:           Int,
    context:         DecisionContext,
    existingOutputs: IndexedSeq[PrefixTree]
  ): TabularDecision = {
    val key = TabularContextKey(context, SearchStateBucket.from(searchState))
    val probabilities = distribution(context, searchState)
    val draw = random.nextDouble()
    var cumulative = 0.0
    var chosen = 0

    while (chosen < probabilities.length - 1) {
      cumulative += probabilities(chosen)
      if (draw < cumulative) {
        return TabularDecision(
          tableKey = key,
          context = context,
          actionIndex = chosen,
          probabilities = probabilities
        )
      }
      chosen += 1
    }

    TabularDecision(
      tableKey = key,
      context = context,
      actionIndex = probabilities.length - 1,
      probabilities = probabilities
    )
  }

  def sample(
    width:           Int,
    context:         DecisionContext,
    existingOutputs: IndexedSeq[PrefixTree]
  ): TabularDecision = sample(SearchPolicyState.empty, width, context, existingOutputs)

  def update(
    trace:        Seq[TabularDecision],
    reward:       Double,
    baseline:     Double,
    learningRate: Double
  ): TabularTrainingStats = {
    require(learningRate > 0.0, s"learningRate must be > 0, got ${learningRate}")

    val advantage = reward - baseline
    trace.foreach { step =>
      val current = logits.getOrElseUpdate(step.tableKey, Vector.fill(step.context.actionCount)(0.0))
      val probs = softmax(current)
      val updated = current.indices.map { idx =>
        val indicator = if (idx == step.actionIndex) 1.0 else 0.0
        current(idx) + learningRate * advantage * (indicator - probs(idx)) / temperature
      }.toVector
      logits.update(step.tableKey, updated)
    }

    val averageEntropy =
      if (trace.isEmpty) 0.0
      else trace.map(_.entropy).sum / trace.length.toDouble

    TabularTrainingStats(
      advantage = advantage,
      averageEntropy = averageEntropy
    )
  }

  def write(path: os.Path): os.Path = {
    os.makeDir.all(path / os.up)
    os.write.over(path, ujson.write(toJson, indent = 2))
    path
  }

  def toJson: ujson.Value = {
    val contexts = logits.toSeq.sortBy(_._1.key).map { case (tableKey, prefs) =>
      ujson.Obj(
        "context" -> ujson.Obj(
          "output_index" -> tableKey.context.outputIndex,
          "segment_low" -> tableKey.context.segmentLow,
          "segment_high" -> tableKey.context.segmentHigh,
          "root" -> tableKey.context.root,
          "action_count" -> tableKey.context.actionCount,
          "key" -> tableKey.context.key
        ),
        "search_bucket" -> ujson.Obj(
          "phase_bucket" -> tableKey.searchBucket.phaseBucket,
          "frontier_bucket" -> tableKey.searchBucket.frontierBucket,
          "duplicate_bucket" -> tableKey.searchBucket.duplicateBucket,
          "spread_bucket" -> tableKey.searchBucket.spreadBucket,
          "key" -> tableKey.searchBucket.key
        ),
        "table_key" -> tableKey.key,
        "logits" -> ujson.Arr.from(prefs.map(ujson.Num(_))),
        "probabilities" -> ujson.Arr.from(softmax(prefs).map(ujson.Num(_)))
      )
    }

    ujson.Obj(
      "policy" -> "tabular-softmax-frontier-bucketed",
      "temperature" -> temperature,
      "context_count" -> logits.size,
      "contexts" -> ujson.Arr.from(contexts)
    )
  }

  private[PrefixTabularRL] def loadFromJson(json: ujson.Value): Unit = {
    logits.clear()
    json.obj.get("contexts") match {
      case Some(ujson.Arr(contexts)) =>
        contexts.foreach { entry =>
          val contextJson = entry("context")
          val bucketJson = entry("search_bucket")
          val context = DecisionContext(
            outputIndex = JsonSupport.readInt(contextJson("output_index")),
            segmentLow = JsonSupport.readInt(contextJson("segment_low")),
            segmentHigh = JsonSupport.readInt(contextJson("segment_high")),
            root = JsonSupport.readBoolean(contextJson("root"))
          )
          val bucket = SearchStateBucket(
            phaseBucket = JsonSupport.readInt(bucketJson("phase_bucket")),
            frontierBucket = JsonSupport.readInt(bucketJson("frontier_bucket")),
            duplicateBucket = JsonSupport.readInt(bucketJson("duplicate_bucket")),
            spreadBucket = JsonSupport.readInt(bucketJson("spread_bucket"))
          )
          val parsedLogits = entry("logits") match {
            case ujson.Arr(values) => values.toVector.map(JsonSupport.readDouble)
            case other             => throw new IllegalArgumentException(s"Expected logits array, found ${other}")
          }
          require(
            parsedLogits.length == context.actionCount,
            s"Loaded logits length ${parsedLogits.length} does not match action count ${context.actionCount} for ${context}"
          )
          logits.update(TabularContextKey(context, bucket), parsedLogits)
        }
      case Some(other) =>
        throw new IllegalArgumentException(s"Expected contexts array in tabular checkpoint, found ${other}")
      case None =>
    }
  }

  private def softmax(values: Vector[Double]): Vector[Double] = {
    val shifted = values.map(_ / temperature)
    val max = shifted.max
    val exp = shifted.map(v => math.exp(v - max))
    val sum = exp.sum
    if (sum == 0.0 || !sum.isFinite) {
      Vector.fill(values.length)(1.0 / values.length.toDouble)
    } else {
      exp.map(_ / sum)
    }
  }
}
