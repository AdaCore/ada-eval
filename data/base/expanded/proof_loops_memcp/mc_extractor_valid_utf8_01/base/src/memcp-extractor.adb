--  Extracted from memcp, src/memcp-extractor.adb.
--
--  Valid_Utf8's body is verbatim upstream, including the two nested expression
--  functions Byte and Cont, which are its meaning and so are copied whole
--  rather than imported.  Only the body-local declaration is gone: the
--  subprogram is now declared in the spec, which carries the doc block.

package body Memcp.Extractor
  with SPARK_Mode => On
is

   ----------------
   -- Valid_Utf8 --
   ----------------

   function Valid_Utf8 (S : String) return Boolean is separate;

end Memcp.Extractor;
