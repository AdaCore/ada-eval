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

   end loop;
end Rebuild_Free_List;
