--
--  Copyright (C) 2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

with SPARK.Big_Intervals; use SPARK.Big_Intervals;
separate (Reachability)
function Model_Reversed
  (X : Extended_Index; M : Memory_Type) return Sequence
is
   Seen     : Memory_Index_Set;
   Not_Seen : Memory_Index_Set := All_Set (M'First, M'Last)
   with Ghost => Static;
   C        : Extended_Index := X;
   R_Seq    : Sequence;
begin
   while C /= No_Index loop
      pragma Loop_Variant (Static => (Decreases => Length (Not_Seen)));
      pragma Loop_Invariant (Static => C in M'Range);
      pragma
        Loop_Invariant
          (Static =>
             (for all I in M'Range =>
                Contains (Seen, I) /= Contains (Not_Seen, I)));
      pragma
        Loop_Invariant
          (Static =>
             (for all I in R_Seq =>
                Get (R_Seq, I)
                = Get
                    (Model_Internal (X, M),
                     Last (Model_Internal (X, M)) - I + 1)));
      pragma
        Loop_Invariant
          (Static =>
             Length (R_Seq)
             <= To_Big (M'Last) - To_Big (M'First) + 1 - Length (Not_Seen));
      pragma
        Loop_Invariant
          (Static =>
             Length (Model_Internal (X, M))
             = Length (R_Seq) + Length (Model_Internal (C, M, Not_Seen)));
      pragma
        Loop_Invariant
          (Static =>
             (for all I in Interval'(1, Last (Model_Internal (X, M))) =>
                Get (Model_Internal (X, M), I)
                = (if I <= Last (Model_Internal (X, M)) - Last (R_Seq)
                   then Get (Model_Internal (C, M, Not_Seen), I)
                   else
                     Get (R_Seq, Last (Model_Internal (X, M)) - I + 1))));
      if Contains (Seen, C) then
         exit;
      end if;
      R_Seq := Add (R_Seq, C);
      Seen := Add (Seen, C);
      Not_Seen := Remove (Not_Seen, C);
      C := Next (M (C));
   end loop;

   return R_Seq;
end Model_Reversed;
