--
--  Copyright (C) 2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (Reachability)
function Is_Acyclic_Impl (X : Extended_Index; M : Memory_Type) return Boolean
is
   Seen     : Memory_Index_Set;
   Not_Seen : Memory_Index_Set := All_Set (M'First, M'Last)
   with Ghost => Static;
   C        : Extended_Index := X;
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
             Is_Acyclic_Internal (C, M, Not_Seen)
             = Is_Acyclic_Internal (X, M));
      if Contains (Seen, C) then
         return False;
      end if;
      Seen := Add (Seen, C);
      Not_Seen := Remove (Not_Seen, C);
      C := Next (M (C));
   end loop;
   return True;
end Is_Acyclic_Impl;
