--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
procedure Assign (Target : in out List; Source : List) is
   N : Node_Array renames Source.Nodes;
   J : Count_Type;

begin
   Clear (Target);

   J := Source.First;
   while J /= 0 loop
      pragma Loop_Invariant (Static => Structural_Invariant (Target));
      pragma
        Loop_Invariant
          (Static => Memory_Index_Sets.Contains (Active_Set (Source), J));
      pragma
        Loop_Invariant
          (Static =>
             Big_Conversions.To_Big (Target.Length)
             + Memory_Index_Sets.Length (Reachable_Set (J, Memory (Source)))
             = Big_Conversions.To_Big (Source.Length));
      pragma
        Loop_Variant
          (Static =>
             (Decreases =>
                Memory_Index_Sets.Length
                  (Reachable_Set (J, Memory (Source)))));

      Lemma_Reachable_Is_Acyclic (Source.First, J, Memory (Source));
      Lemma_Reachable_Included (Source.First, J, Memory (Source));

      --  Advancing J to its successor drops it from the reachable set, so
      --  the count decreases by one (matching the node Append adds).

      Lemma_Reachable_Def (J, Memory (Source));
      Lemma_Is_Acyclic_Def (J, Memory (Source));

      Append (Target, N (J).Element, 1);
      J := N (J).Next;
   end loop;
end Assign;
