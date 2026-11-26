--
--  Copyright (C) 2022-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Warnings (Off, "unused variable ""E""");
pragma Ada_2022;
separate (SPARK.Containers.Functional.Multisets)
function Union (Left : Multiset; Right : Multiset) return Multiset is
   Set    : Multiset := Left;
   Buffer : Multiset := Right;
   E      : Element_Type;
   Occ    : Big_Integer;

begin
   while not Is_Empty (Buffer) loop
      E := Choose (Buffer);
      Buffer := Remove_All (Buffer, E);

      Occ := Max (Nb_Occurence (Left, E), Nb_Occurence (Right, E));
      if Occ > Nb_Occurence (Set, E) then
         Set := Add (Set, E, Occ - Nb_Occurence (Set, E));
      end if;

      pragma Loop_Variant (Decreases => Cardinality (Buffer));

   end loop;
   Lemma_Leq_Cardinality (Right.Map, Set.Map);
   return Set;
end Union;
