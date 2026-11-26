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

      pragma
        Assert
          (Static =>
             (for all Element of Set =>
                (if Equivalent_Elements (Element, E)
                 then
                   Nb_Occurence (Set, Element)
                   <= Nb_Occurence (Left, Element))));
      pragma
        Assert
          (Static =>
             (for all Element of Set => Has_Key (Left.Map, Element)));
      Lemma_Leq_Cardinality (Set.Map, Left.Map);
      pragma Assert (Static => Cardinality (Set) <= Cardinality (Left));

      pragma
        Assert
          (Static =>
             (for all Element of Set =>
                Has_Key (Right.Map, Element)
                and then
                  Get (Set.Map, Element) <= Get (Right.Map, Element)));
      Lemma_Leq_Cardinality (Set.Map, Right.Map);

      pragma Loop_Variant (Decreases => Cardinality (Buffer));

      pragma Loop_Invariant (Static => Invariant (Buffer.Map, Buffer.Card));
      pragma Loop_Invariant (Static => Invariant (Set.Map, Set.Card));
      pragma Loop_Invariant (Static => Set <= Left);
      pragma Loop_Invariant (Static => Set <= Right);
      pragma
        Loop_Invariant
          (Static =>
             (for all Element of Buffer =>
                Nb_Occurence (Set, Element) = 0));

      pragma
        Loop_Invariant
          (Static =>
             (for all Element of Left =>
                (if not Contains (Buffer, Element)
                 then
                   Min
                     (Nb_Occurence (Left, Element),
                      Nb_Occurence (Right, Element))
                   = Nb_Occurence (Set, Element))));

   end loop;
   return Set;
end Intersection;
