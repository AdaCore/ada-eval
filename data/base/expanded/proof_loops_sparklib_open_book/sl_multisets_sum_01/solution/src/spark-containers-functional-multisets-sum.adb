--
--  Copyright (C) 2022-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Warnings (Off, "procedure ""Lemma_Leq_Cardinality"" is not referenced");
pragma Warnings (Off, "unused variable ""E""");
pragma Ada_2022;
separate (SPARK.Containers.Functional.Multisets)
function Sum (Left : Multiset; Right : Multiset) return Multiset is
   E         : Element_Type;
   Set_Union : Multiset := Left;
   Buffer    : Multiset;

begin
   Buffer := Right;
   while not Is_Empty (Buffer) loop
      E := Choose (Buffer);

      Set_Union := Add (Set_Union, E, Get (Buffer.Map, E));
      Buffer := Remove_All (Buffer, E);

      pragma Loop_Variant (Decreases => Cardinality (Buffer));
      pragma
        Loop_Invariant
          (Static => Invariant (Set_Union.Map, Set_Union.Card));
      pragma Loop_Invariant (Static => Invariant (Buffer.Map, Buffer.Card));
      pragma
        Loop_Invariant
          (Static =>
             Cardinality (Set_Union) + Cardinality (Buffer)
             = Cardinality (Left) + Cardinality (Right));
      pragma
        Loop_Invariant
          (Static =>
             (for all Element of Right =>
                Contains (Set_Union, Element)
                or else Contains (Buffer, Element)));
      pragma
        Loop_Invariant
          (Static =>
             (for all Element of Left => Contains (Set_Union, Element)));
      pragma
        Loop_Invariant
          (Static =>
             (for all Element of Buffer =>
                Nb_Occurence (Set_Union, Element)
                = Nb_Occurence (Left, Element)));
      pragma
        Loop_Invariant
          (Static =>
             (for all Element of Buffer =>
                Nb_Occurence (Buffer, Element)
                = Nb_Occurence (Right, Element)));
      pragma
        Loop_Invariant
          (Static =>
             (for all Element of Set_Union =>
                (if Contains (Buffer, Element)
                 then
                   Nb_Occurence (Set_Union, Element)
                   = Nb_Occurence (Left, Element)
                 else
                   Nb_Occurence (Set_Union, Element)
                   = Nb_Occurence (Left, Element)
                     + Nb_Occurence (Right, Element))));
   end loop;

   return Set_Union;
end Sum;
