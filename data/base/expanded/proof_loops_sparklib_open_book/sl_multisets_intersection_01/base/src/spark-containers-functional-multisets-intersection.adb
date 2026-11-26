--
--  Copyright (C) 2022-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Warnings (Off, "unused variable ""E""");
pragma Ada_2022;
separate (SPARK.Containers.Functional.Multisets)
function Intersection (Left : Multiset; Right : Multiset) return Multiset is
   Set    : Multiset;
   Buffer : Multiset := Left;
   E      : Element_Type;
   Occ    : Big_Integer;

begin
   while not Is_Empty (Buffer) loop
      E := Choose (Buffer);
      Buffer := Remove_All (Buffer, E);

      Occ := Min (Nb_Occurence (Left, E), Nb_Occurence (Right, E));
      if Occ > 0 then
         Set := Add (Set, E, Occ);
      end if;

      Lemma_Leq_Cardinality (Set.Map, Left.Map);

      Lemma_Leq_Cardinality (Set.Map, Right.Map);

      pragma Loop_Variant (Decreases => Cardinality (Buffer));

   end loop;
   return Set;
end Intersection;
