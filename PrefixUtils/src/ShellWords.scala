package PrefixUtils

import scala.collection.mutable

object ShellWords {
  def split(raw: String): Vector[String] = {
    val tokens = mutable.ArrayBuffer.empty[String]
    val current = new StringBuilder

    var inSingleQuotes = false
    var inDoubleQuotes = false
    var escaping = false
    var tokenStarted = false

    def commit(): Unit = {
      if (tokenStarted) {
        tokens += current.toString()
        current.clear()
        tokenStarted = false
      }
    }

    raw.foreach { ch =>
      if (escaping) {
        current.append(ch)
        tokenStarted = true
        escaping = false
      } else if (inSingleQuotes) {
        if (ch == '\'') {
          inSingleQuotes = false
        } else {
          current.append(ch)
          tokenStarted = true
        }
      } else if (inDoubleQuotes) {
        ch match {
          case '"' =>
            inDoubleQuotes = false
          case '\\' =>
            escaping = true
          case other =>
            current.append(other)
            tokenStarted = true
        }
      } else {
        ch match {
          case other if other.isWhitespace =>
            commit()
          case '\'' =>
            inSingleQuotes = true
            tokenStarted = true
          case '"' =>
            inDoubleQuotes = true
            tokenStarted = true
          case '\\' =>
            escaping = true
            tokenStarted = true
          case other =>
            current.append(other)
            tokenStarted = true
        }
      }
    }

    if (escaping) {
      throw new IllegalArgumentException(s"Unterminated escape sequence in command: '${raw}'")
    }
    if (inSingleQuotes || inDoubleQuotes) {
      throw new IllegalArgumentException(s"Unterminated quoted string in command: '${raw}'")
    }

    commit()
    tokens.toVector
  }
}
