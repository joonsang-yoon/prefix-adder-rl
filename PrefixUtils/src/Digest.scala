package PrefixUtils

import java.nio.charset.StandardCharsets
import java.security.MessageDigest

object Digest {
  def sha1(text: String): String = {
    val md = MessageDigest.getInstance("SHA-1")
    val bytes = md.digest(text.getBytes(StandardCharsets.UTF_8))
    bytes.map(b => f"${b & 0xff}%02x").mkString
  }
}
