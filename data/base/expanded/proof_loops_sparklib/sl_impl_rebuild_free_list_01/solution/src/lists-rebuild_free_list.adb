--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
procedure Rebuild_Free_List (L : List; Base, Cont : Count_Type) is
   M : constant Nodes_Type_Base := Memory (L)
   with Ghost => Static;
begin
   for J in reverse Base .. L.Capacity loop
      Lemma_Is_Acyclic_Def (J, M);
      Lemma_Reachable_Def (J, M);

      pragma Loop_Invariant (Static => Is_Acyclic (J, M));
      pragma
        Loop_Invariant
          (Static =>
             (for all I of Reachable_Set (J, M) =>
                I in J .. L.Capacity
                or else
                  Memory_Index_Sets.Contains
                    (Reachable_Set (Cont, M), I)));
      pragma
        Loop_Invariant
          (Static =>
             (for all I of Reachable_Set (J, M) =>
                L.Nodes (I).Prev = -1));
      pragma
        Loop_Invariant
          (Static =>
             (for all I in J .. L.Capacity =>
                Memory_Index_Sets.Contains (Reachable_Set (J, M), I)));
      pragma
        Loop_Invariant
          (Static =>
             (for all I of Reachable_Set (Cont, M) =>
                Memory_Index_Sets.Contains (Reachable_Set (J, M), I)));
      pragma
        Loop_Invariant
          (Static =>
             Memory_Index_Sets.Length (Reachable_Set (J, M))
             = Big_Conversions.To_Big (L.Capacity - J + 1)
               + Memory_Index_Sets.Length (Reachable_Set (Cont, M)));
   end loop;
end Rebuild_Free_List;
