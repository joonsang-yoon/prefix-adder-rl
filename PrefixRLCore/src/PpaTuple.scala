package PrefixRLCore

final case class PpaTuple(power: Double, delay: Double, area: Double) {
  require(power.isFinite && delay.isFinite && area.isFinite, s"PPA tuple must be finite, found ${this}")

  def dominates(other: PpaTuple): Boolean = {
    val noWorse = power <= other.power && delay <= other.delay && area <= other.area
    val strictlyBetter = power < other.power || delay < other.delay || area < other.area
    noWorse && strictlyBetter
  }

  def toJson: ujson.Value = ujson.Obj(
    "power" -> power,
    "delay" -> delay,
    "area" -> area
  )
}

final case class NormalizedObjectives(power: Double, delay: Double, area: Double) {
  def average: Double = (power + delay + area) / 3.0

  def distanceTo(other: NormalizedObjectives): Double = {
    val dp = power - other.power
    val dd = delay - other.delay
    val da = area - other.area
    math.sqrt(dp * dp + dd * dd + da * da)
  }

  def toJson: ujson.Value = ujson.Obj(
    "power" -> power,
    "delay" -> delay,
    "area" -> area,
    "average" -> average
  )
}

final class RunningObjectiveNormalizer {
  private var minPower = Double.PositiveInfinity
  private var minDelay = Double.PositiveInfinity
  private var minArea = Double.PositiveInfinity

  private var maxPower = Double.NegativeInfinity
  private var maxDelay = Double.NegativeInfinity
  private var maxArea = Double.NegativeInfinity

  def observe(ppa: PpaTuple): Unit = {
    minPower = math.min(minPower, ppa.power)
    minDelay = math.min(minDelay, ppa.delay)
    minArea = math.min(minArea, ppa.area)

    maxPower = math.max(maxPower, ppa.power)
    maxDelay = math.max(maxDelay, ppa.delay)
    maxArea = math.max(maxArea, ppa.area)
  }

  def normalize(ppa: PpaTuple): NormalizedObjectives = {
    NormalizedObjectives(
      power = normalizeOne(ppa.power, minPower, maxPower),
      delay = normalizeOne(ppa.delay, minDelay, maxDelay),
      area = normalizeOne(ppa.area, minArea, maxArea)
    )
  }

  private def normalizeOne(value: Double, minValue: Double, maxValue: Double): Double = {
    if (!minValue.isFinite || !maxValue.isFinite) {
      0.5
    } else {
      val span = maxValue - minValue
      if (math.abs(span) < 1e-12) {
        0.0
      } else {
        ((value - minValue) / span).max(0.0).min(1.0)
      }
    }
  }
}
