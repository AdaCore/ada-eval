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

   end loop;

   --  Head < Base, so Head is not already on the tail chain; prepending
   --  it keeps the chain acyclic and adds exactly one node.

   Lemma_Is_Acyclic_Def (Head, M);
   Lemma_Reachable_Def (Head, M);
end Rebuild_Free_List;
