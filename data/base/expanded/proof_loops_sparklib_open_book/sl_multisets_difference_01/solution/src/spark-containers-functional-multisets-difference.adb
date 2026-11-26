--
--  Copyright (C) 2022-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Warnings (Off, "procedure ""Lemma_Leq_Cardinality"" is not referenced");
pragma Warnings (Off, "unused variable ""E""");
pragma Ada_2022;
separate (SPARK.Containers.Functional.Multisets)
function Difference (Left : Multiset; Right : Multiset) return Multiset is
   Set     : Multiset;
   Count   : Big_Integer;
   Buffer  : Multiset := Left;
   Element : Element_Type;

begin

   while not Is_Empty (Buffer) loop
      Element := Choose (Buffer);
      Buffer := Remove_All (Buffer, Element);

      Count := Nb_Occurence (Left, Element) - Nb_Occurence (Right, Element);
      if Count > 0 then
         Set := Add (Set, Element, Count);
      end if;

      pragma Loop_Variant (Decreases => Cardinality (Buffer));

      pragma Loop_Invariant (Static => Invariant (Buffer.Map, Buffer.Card));
      pragma Loop_Invariant (Static => Invariant (Set.Map, Set.Card));
      pragma
        Loop_Invariant
          (Static => (for all E of Buffer => Nb_Occurence (Set, E) = 0));
      pragma
        Loop_Invariant
          (Static =>
             (for all E of Set =>
                Nb_Occurence (Set, E)
                = Nb_Occurence (Left, E) - Nb_Occurence (Right, E)));
      pragma
        Loop_Invariant
          (Static =>
             (for all E of Left =>
                (if not Contains (Buffer, E)
                   and then
                     Nb_Occurence (Left, E) - Nb_Occurence (Right, E) > 0
                 then Contains (Set, E))));

   end loop;
   return Set;
end Difference;
