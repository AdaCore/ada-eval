--
--  Copyright (C) 2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (Reachability)
function Model_From_Reversed
  (X : Extended_Index; M : Memory_Type; R_Seq : Sequence) return Sequence is
begin

   --  Reverse the sequence

   return Seq : Sequence do
      declare
         I : Big_Integer := Last (R_Seq);
      begin
         while I > 0 loop
            pragma Loop_Variant (Static => (Decreases => I));
            pragma
              Loop_Invariant
                (Static =>
                   Length (Seq) = Length (Model_Internal (X, M)) - I);
            pragma
              Loop_Invariant
                (Static =>
                   (for all I in Seq =>
                      Get (Seq, I) = Get (Model_Internal (X, M), I)));
            Seq := Add (Seq, Get (R_Seq, I));
            I := I - 1;
         end loop;
      end;
   end return;
end Model_From_Reversed;
