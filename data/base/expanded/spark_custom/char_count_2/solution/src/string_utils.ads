package String_Utils is

   function Count_Ghost
     (Str : String; Char : Character; Idx : Integer) return Natural
   is (if Idx not in Str'Range
       then 0
       else
         (if Str (Idx) = Char
          then 1 + Count_Ghost (Str, Char, Idx - 1)
          else 0 + Count_Ghost (Str, Char, Idx - 1)))
   with
     Ghost,
     Post               =>
       (if Idx not in Str'Range
        then Count_Ghost'Result = 0
        else Count_Ghost'Result <= (Idx - Str'First) + 1),
     Subprogram_Variant => (Decreases => Idx);

   function Count (Str : String; Char : Character) return Natural
   with Post => Count'Result = Count_Ghost (Str, Char, Str'Last);

end String_Utils;
