--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
procedure Rebuild_Free_List (C : List; Head, Base : Count_Type) is
   M : constant Nodes_Type_Base := Memory (C)
   with Ghost => Static;
begin
   for J in reverse Base .. C.Capacity loop
      Lemma_Is_Acyclic_Def (J, M);
      Lemma_Reachable_Def (J, M);

      pragma Loop_Invariant (Static => Is_Acyclic (J, M));
      pragma
        Loop_Invariant
          (Static =>
             (for all I of Reachable_Set (J, M) =>
                I in J .. C.Capacity));
      pragma
        Loop_Invariant
          (Static =>
             (for all I in J .. C.Capacity =>
                Memory_Index_Sets.Contains (Reachable_Set (J, M), I)));
      pragma
        Loop_Invariant
          (Static =>
             Memory_Index_Sets.Length (Reachable_Set (J, M))
             = Big_Conversions.To_Big (C.Capacity - J + 1));
   end loop;

   --  Head < Base, so Head is not already on the tail chain; prepending
   --  it keeps the chain acyclic and adds exactly one node.

   Lemma_Is_Acyclic_Def (Head, M);
   Lemma_Reachable_Def (Head, M);
end Rebuild_Free_List;
