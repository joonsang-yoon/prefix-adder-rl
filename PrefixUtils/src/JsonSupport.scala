package PrefixUtils

object JsonSupport {
  def readDouble(value: ujson.Value): Double = value match {
    case ujson.Num(v) => v
    case ujson.Str(v) => v.toDouble
    case other        => throw new IllegalArgumentException(s"Expected numeric JSON value, found ${other}")
  }

  def readInt(value: ujson.Value): Int = value match {
    case ujson.Num(v) =>
      if (!v.isWhole || !v.isValidInt) {
        throw new IllegalArgumentException(s"Expected Int-valued JSON number, found ${v}")
      }
      v.toInt

    case ujson.Str(v) =>
      val parsed =
        try BigInt(v.trim)
        catch {
          case _: NumberFormatException =>
            throw new IllegalArgumentException(s"Expected integer JSON string, found '${v}'")
        }

      if (!parsed.isValidInt) {
        throw new IllegalArgumentException(s"Integer JSON value out of Int range: '${v}'")
      }
      parsed.toInt

    case other =>
      throw new IllegalArgumentException(s"Expected integer JSON value, found ${other}")
  }

  def readLong(value: ujson.Value): Long = value match {
    case ujson.Num(v) =>
      if (
        !v.isFinite ||
        !v.isWhole ||
        v < Long.MinValue.toDouble ||
        v > Long.MaxValue.toDouble
      ) {
        throw new IllegalArgumentException(s"Expected Long-valued JSON number, found ${v}")
      }
      v.toLong

    case ujson.Str(v) =>
      val parsed =
        try BigInt(v.trim)
        catch {
          case _: NumberFormatException =>
            throw new IllegalArgumentException(s"Expected long JSON string, found '${v}'")
        }

      if (!parsed.isValidLong) {
        throw new IllegalArgumentException(s"Long JSON value out of range: '${v}'")
      }
      parsed.toLong

    case other =>
      throw new IllegalArgumentException(s"Expected long JSON value, found ${other}")
  }

  def readBoolean(value: ujson.Value): Boolean = value match {
    case ujson.Bool(v) => v
    case ujson.Str(v) =>
      v.trim.toLowerCase match {
        case "true"  => true
        case "false" => false
        case other   => throw new IllegalArgumentException(s"Expected boolean JSON string, found '${other}'")
      }
    case other =>
      throw new IllegalArgumentException(s"Expected boolean JSON value, found ${other}")
  }

  def readString(value: ujson.Value): String = value match {
    case ujson.Str(v) => v
    case other        => throw new IllegalArgumentException(s"Expected string JSON value, found ${other}")
  }

  def writeFile(path: os.Path, content: String): Unit = {
    if (!os.exists(path / os.up)) {
      os.makeDir.all(path / os.up)
    }
    os.write.over(path, content)
  }
}
