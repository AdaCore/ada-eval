package Proof_Harness is

   function Ref_Count
     (Str : String; Char : Character; Idx : Integer) return Natural
   is (if Idx not in Str'Range
       then 0
       else
         (if Str (Idx) = Char then 1 else 0) + Ref_Count (Str, Char, Idx - 1))
   with
     Ghost,
     Post               =>
       (if Idx not in Str'Range
        then Ref_Count'Result = 0
        else Ref_Count'Result <= (Idx - Str'First) + 1),
     Subprogram_Variant => (Decreases => Idx);

   function Check_Count (Str : String; Char : Character) return Natural
   with Post => Check_Count'Result = Ref_Count (Str, Char, Str'Last);

end Proof_Harness;
