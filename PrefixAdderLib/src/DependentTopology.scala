package PrefixAdderLib

import PrefixUtils.{Digest, JsonSupport}

import scala.collection.mutable

sealed trait PrefixTree {
  def low:  Int
  def high: Int

  final def depth: Int = this match {
    case Leaf(_)           => 0
    case Node(left, right) => 1 + math.max(left.depth, right.depth)
  }

  final def leafCount: Int = high - low + 1

  final def internalNodeCount: Int = this match {
    case Leaf(_)           => 0
    case Node(left, right) => 1 + left.internalNodeCount + right.internalNodeCount
  }

  final def isOrderedContiguous: Boolean = this match {
    case Leaf(_) => true
    case Node(left, right) =>
      left.isOrderedContiguous && right.isOrderedContiguous && left.high + 1 == right.low
  }

  final def signature: String = this match {
    case Leaf(index)       => s"L${index}"
    case Node(left, right) => s"N(${left.signature},${right.signature})"
  }

  final def pretty: String = this match {
    case Leaf(index)       => s"x${index}"
    case Node(left, right) => s"(${left.pretty} ∘ ${right.pretty})"
  }

  final def toJson: ujson.Value = this match {
    case Leaf(index) => ujson.Obj("leaf" -> index)
    case Node(left, right) =>
      ujson.Obj(
        "node" -> ujson.Arr(left.toJson, right.toJson)
      )
  }
}

final case class Leaf(index: Int) extends PrefixTree {
  override val low:  Int = index
  override val high: Int = index
}

final case class Node(left: PrefixTree, right: PrefixTree) extends PrefixTree {
  override val low:  Int = left.low
  override val high: Int = right.high
}

final case class TopologyStats(
  uniqueInternalNodes: Int,
  totalInternalNodes:  Int,
  maxDepth:            Int,
  averageDepth:        Double,
  fingerprint:         String
) {
  def reuseRatio: Double = {
    if (totalInternalNodes == 0) {
      1.0
    } else {
      1.0 - uniqueInternalNodes.toDouble / totalInternalNodes.toDouble
    }
  }
}

final case class DependentTopology(width: Int, outputs: Vector[PrefixTree]) {
  require(width >= 1, s"width must be >= 1, got ${width}")
  require(outputs.length == width, s"expected ${width} outputs, got ${outputs.length}")

  validate()

  lazy val stats: TopologyStats = {
    val unique = mutable.LinkedHashSet.empty[String]
    outputs.foreach(tree => collectInternalNodeSignatures(tree, unique))
    val total = outputs.map(_.internalNodeCount).sum
    val maxDepth = outputs.map(_.depth).max
    val avgDepth = outputs.map(_.depth.toDouble).sum / outputs.length.toDouble
    TopologyStats(
      uniqueInternalNodes = unique.size,
      totalInternalNodes = total,
      maxDepth = maxDepth,
      averageDepth = avgDepth,
      fingerprint = Digest.sha1(toCompactJson).take(16)
    )
  }

  def toJson: ujson.Value = ujson.Obj(
    "model" -> "dependent-tree",
    "width" -> width,
    "outputs" -> ujson.Arr.from(outputs.map(_.toJson))
  )

  def toCompactJson: String = ujson.write(toJson, indent = 0)

  def toPrettyJson: String = ujson.write(toJson, indent = 2)

  def write(path: os.Path): os.Path = {
    os.makeDir.all(path / os.up)
    os.write.over(path, toPrettyJson)
    path
  }

  def toDot: String = {
    val lines = mutable.ArrayBuffer.empty[String]
    lines += "digraph DependentPrefixTopology {"
    lines += "  rankdir=LR;"
    lines += "  node [shape=box, fontname=Helvetica];"

    val ids = mutable.HashMap.empty[String, String]
    var nextId = 0

    def emit(tree: PrefixTree): String = {
      val sig = tree.signature
      ids.getOrElseUpdate(
        sig, {
          val id = s"n${nextId}"
          nextId += 1
          val label = tree match {
            case Leaf(index)       => s"x${index}"
            case Node(left, right) => s"[${tree.low},${tree.high}]"
          }
          lines += s"  ${id} [label=\"${label}\"];"
          tree match {
            case Leaf(_) =>
            case Node(left, right) =>
              val l = emit(left)
              val r = emit(right)
              lines += s"  ${id} -> ${l};"
              lines += s"  ${id} -> ${r};"
          }
          id
        }
      )
    }

    outputs.zipWithIndex.foreach { case (tree, idx) =>
      val rootId = emit(tree)
      lines += s"  out${idx} [shape=oval, label=\"P${idx}\"];"
      lines += s"  out${idx} -> ${rootId};"
    }

    lines += "}"
    lines.mkString("\n")
  }

  private def validate(): Unit = {
    require(outputs.head == Leaf(0), s"P0 must be exactly Leaf(0), found ${outputs.head.pretty}")

    outputs.zipWithIndex.foreach { case (tree, idx) =>
      require(tree.low == 0, s"output P${idx} must start at bit 0, found low=${tree.low}")
      require(tree.high == idx, s"output P${idx} must end at bit ${idx}, found high=${tree.high}")
      require(tree.isOrderedContiguous, s"output P${idx} is not an ordered contiguous tree")

      if (idx > 0) {
        tree match {
          case Node(left, right) =>
            val split = right.low - 1
            require(split >= 0 && split < idx, s"output P${idx} has invalid dependent split ${split}")
            require(
              left == outputs(split),
              s"output P${idx} violates dependent-tree construction: left subtree is not P${split}"
            )
          case Leaf(_) =>
            throw new IllegalArgumentException(s"output P${idx} must be a Node in the dependent-tree model")
        }
      }
    }
  }

  private def collectInternalNodeSignatures(tree: PrefixTree, acc: mutable.Set[String]): Unit = tree match {
    case Leaf(_) =>
    case node @ Node(left, right) =>
      acc += node.signature
      collectInternalNodeSignatures(left, acc)
      collectInternalNodeSignatures(right, acc)
  }
}

object DependentTopology {
  def fromJsonString(text: String): DependentTopology = {
    val value = ujson.read(text)
    val model = value.obj.get("model").map(_.str).getOrElse("dependent-tree")
    require(model == "dependent-tree", s"Only the dependent-tree model is supported, found '${model}'")
    val width = JsonSupport.readInt(value("width"))
    val outputs = value("outputs").arr.map(fromJsonTree).toVector
    DependentTopology(width, outputs)
  }

  def fromFile(path: os.Path): DependentTopology = fromJsonString(os.read(path))

  def ripple(width: Int): DependentTopology = {
    require(width >= 1, s"width must be >= 1, got ${width}")
    val built = mutable.ArrayBuffer.empty[PrefixTree]
    built += Leaf(0)
    for (idx <- 1 until width) {
      built += Node(built.last, Leaf(idx))
    }
    DependentTopology(width, built.toVector)
  }

  def balanced(width: Int): DependentTopology = {
    require(width >= 1, s"width must be >= 1, got ${width}")
    val built = mutable.ArrayBuffer.empty[PrefixTree]
    built += Leaf(0)
    for (idx <- 1 until width) {
      val split = idx / 2
      val right = buildBalancedSuffix(split + 1, idx)
      built += Node(built(split), right)
    }
    DependentTopology(width, built.toVector)
  }

  private def buildBalancedSuffix(low: Int, high: Int): PrefixTree = {
    require(low <= high)
    if (low == high) {
      Leaf(low)
    } else {
      val split = (low + high) / 2
      Node(buildBalancedSuffix(low, split), buildBalancedSuffix(split + 1, high))
    }
  }

  private def fromJsonTree(value: ujson.Value): PrefixTree = {
    val obj = value.obj

    (obj.get("leaf"), obj.get("node")) match {
      case (Some(index), None) =>
        Leaf(JsonSupport.readInt(index))

      case (None, Some(nodeValue)) =>
        val arr = nodeValue.arr
        require(arr.length == 2, s"node must contain exactly two children, found ${arr.length}")
        Node(fromJsonTree(arr(0)), fromJsonTree(arr(1)))

      case (Some(_), Some(_)) =>
        throw new IllegalArgumentException(
          s"tree node must contain exactly one of 'leaf' or 'node', found both: ${value}"
        )

      case (None, None) =>
        throw new IllegalArgumentException(
          s"tree node must contain either 'leaf' or 'node': ${value}"
        )
    }
  }
}
