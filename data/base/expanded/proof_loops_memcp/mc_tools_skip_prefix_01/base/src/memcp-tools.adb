package body Memcp.Tools
  with SPARK_Mode => On
is

   function After (S : String; E : Positive) return Natural
   is (if E < S'Last then E + 1 else 0)
   with
     Pre  => E in S'Range,
     Post => After'Result = 0 or else After'Result in S'Range;
   --  The index just past E, or 0 when E is the last character: an end
   --  sentinel
   --  that keeps every cursor a valid index and never forms S'Last + 1.

   -----------------
   -- Skip_Prefix --
   -----------------

   function Skip_Prefix (S : String; P : Positive) return Natural is separate;

end Memcp.Tools;
