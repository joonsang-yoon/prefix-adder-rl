package PrefixUtils

object Catalan {
  private val memo = scala.collection.mutable.HashMap.empty[Int, BigInt]

  def apply(n: Int): BigInt = memo.getOrElseUpdate(n, catalanClosedForm(n))

  def sequence(n: Int): Vector[BigInt] = Vector.tabulate(n + 1)(apply)

  def extensionCount(width: Int): BigInt = {
    require(width >= 1, s"width must be >= 1, got ${width}")
    if (width == 1) {
      BigInt(1)
    } else {
      (0 to width - 2).foldLeft(BigInt(0)) { case (acc, idx) => acc + apply(idx) }
    }
  }

  def dependentNetworkCount(width: Int): BigInt = {
    require(width >= 1, s"width must be >= 1, got ${width}")
    (2 to width).foldLeft(BigInt(1)) { case (acc, n) =>
      acc * extensionCount(n)
    }
  }

  private def catalanClosedForm(n: Int): BigInt = {
    require(n >= 0, s"Catalan index must be >= 0, got ${n}")
    if (n == 0) {
      BigInt(1)
    } else {
      binomial(2 * n, n) / (n + 1)
    }
  }

  private def binomial(n: Int, k: Int): BigInt = {
    val kk = math.min(k, n - k)
    (1 to kk).foldLeft(BigInt(1)) { case (acc, i) =>
      acc * (n - kk + i) / i
    }
  }
}
