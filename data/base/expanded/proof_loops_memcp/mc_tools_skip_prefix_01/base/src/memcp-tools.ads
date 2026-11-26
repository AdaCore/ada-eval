--  Extracted from memcp, src/memcp-tools.adb.
--
--  Skip_Prefix is one step of the close-tag scanner Memcp.Tools uses to repair
--  a `save` call whose Header and Summary arrived as a single blob: given a
--  cursor P into the text, it skips an optional XML-style `ns:` namespace
--  prefix so that the tag vocabulary can be matched at the name proper.

package Memcp.Tools
  with SPARK_Mode => On
is

   function Skip_Prefix (S : String; P : Positive) return Natural
   with
     Pre  => P in S'Range,
     Post => Skip_Prefix'Result = 0 or else Skip_Prefix'Result in S'Range;
   --  Index of the first tag-name character at P, skipping an optional `ns:`
   --  prefix -- a letter, then name characters, then a colon. P itself when
   --  there is no prefix; 0 when the prefix runs to the end of S.

end Memcp.Tools;
