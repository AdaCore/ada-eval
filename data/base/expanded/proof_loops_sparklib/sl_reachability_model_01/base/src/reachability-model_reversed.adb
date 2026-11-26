--
--  Copyright (C) 2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

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
