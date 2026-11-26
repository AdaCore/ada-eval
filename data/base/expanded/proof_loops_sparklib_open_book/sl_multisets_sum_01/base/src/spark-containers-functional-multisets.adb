--
--  Copyright (C) 2022-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Warnings (Off, "procedure ""Lemma_Leq_Cardinality"" is not referenced");
pragma Warnings (Off, "unused variable ""E""");
pragma Ada_2022;

package body SPARK.Containers.Functional.Multisets is

   -----------------------
   -- Local Subprograms --
   -----------------------

   function Cardinality_Rec (Container : Maps.Map) return Big_Natural
   with
     --  Computes recursively the Cardinality - i.e. the sum of the number of
     --  occurences of all the elements - of Container.

     Ghost              => SPARKlib_Full,
     Global             => null,
     Subprogram_Variant => (Decreases => Maps.Length (Container)),
     Post               =>
       (for all Element of Container =>
          Get (Container, Element) <= Cardinality_Rec'Result);

   ------------------
   -- Local Lemmas --
   ------------------

   procedure Lemma_Eq (M1, M2 : Maps.Map)
   with
     --  States that if two multisets are equal, then they have the same
     --  cardinality.

     Ghost              => Static,
     Pre                => M1 = M2,
     Post               => Cardinality_Rec (M1) = Cardinality_Rec (M2),
     Subprogram_Variant => (Decreases => Length (M1), Decreases => 1);

   procedure Lemma_Remove (M : Maps.Map; K : Element_Type)
   with
     Ghost              => Static,
     Subprogram_Variant => (Decreases => Maps.Length (M), Decreases => 0),
     Pre                => Has_Key (M, K),
     Post               =>
       Cardinality_Rec (M) = Cardinality_Rec (Remove (M, K)) + Get (M, K);
   --  States that if an element is removed from a multiset, then its number of
   --  occurences is removed from the cardinality of this multiset.

   procedure Lemma_Empty (M : Multiset)
   with
     --  This lemma eases the proof of Is_Empty bridging the gap between the
     --  Is_Empty function on functional maps and the one on multisets.

     Ghost => Static,
     Pre   => Invariant (M.Map, M.Card),
     Post  => Is_Empty (M.Map) = (for all E of M => False);

   procedure Lemma_Contains_Cardinality
     (Container : Multiset; Element : Element_Type)
   with
     --  States that the cardinality of a multiset is larger than the number of
     --  occurences of each of its elements.

     Ghost  => Static,
     Global => null,
     Pre    =>
       Invariant (Container.Map, Container.Card)
       and then Contains (Container, Element),
     Post   => Nb_Occurence (Container, Element) <= Cardinality (Container);

   procedure Lemma_Leq_Cardinality (Left, Right : Maps.Map)
   with
     Ghost              => Static,
     Subprogram_Variant => (Decreases => Length (Left)),
     Pre                =>
       (for all E of Left =>
          Has_Key (Right, E) and then Get (Left, E) <= Get (Right, E)),
     Post               => Cardinality_Rec (Left) <= Cardinality_Rec (Right);
   --  States that if Left contains less occurences of each element value than
   --  Right, then the cardinality of Left is smaller than the cardinality of
   --  Right.

   ---------
   -- Add --
   ---------

   function Add
     (Container : Multiset; Element : Element_Type; Count : Big_Positive)
      return Multiset is
   begin
      return M : Multiset do
         if Count > 0 then
            if Has_Key (Container.Map, Element) then
               M.Map :=
                 Set
                   (Container.Map,
                    Element,
                    Get (Container.Map, Element) + Count);
               Lemma_Eq
                 (Remove (M.Map, Element), Remove (Container.Map, Element));
               Lemma_Remove (Container.Map, Element);
            else
               M.Map := Add (Container.Map, Element, Count);
               Lemma_Eq (Remove (M.Map, Element), Container.Map);
            end if;
            M.Card := Container.Card + Count;
            Lemma_Remove (M.Map, Element);
         end if;
      end return;
   end Add;

   function Add (Container : Multiset; Element : Element_Type) return Multiset
   is
   begin
      return Add (Container, Element, 1);
   end Add;

   ------------------
   -- Aggr_Include --
   ------------------

   procedure Aggr_Include
     (Container : in out Multiset; Element : Element_Type; Count : Big_Natural)
   is
   begin
      if Count /= 0 then
         Container := Add (Container, Element, Count);
      end if;
   end Aggr_Include;

   -----------------
   -- Cardinality --
   -----------------

   function Cardinality (Container : Multiset) return Big_Natural
   is (Container.Card);

   ---------------------
   -- Cardinality_Rec --
   ---------------------

   function Cardinality_Rec (Container : Maps.Map) return Big_Natural
   is (if Length (Container) = 0
       then 0
       else
         Get (Container, Choose (Container))
         + Cardinality_Rec (Remove (Container, Choose (Container))));

   --------------
   -- Contains --
   --------------

   function Contains
     (Container : Multiset; Element : Element_Type) return Boolean is
   begin
      return Has_Key (Container.Map, Element);
   end Contains;

   ------------
   -- Choose --
   ------------

   function Choose (Container : Multiset) return Element_Type
   is (Choose (Container.Map));

   --------------------
   -- Empty_Multiset --
   --------------------

   function Empty_Multiset return Multiset is
   begin
      return (Map => Empty_Map, Card => 0);
   end Empty_Multiset;

   ---------------
   -- Invariant --
   ---------------

   function Invariant (Container : Map; Card : Big_Natural) return Boolean
   is (if Length (Container) = 0
       then Card = 0
       else Card = Cardinality_Rec (Container));

   --------------
   -- Is_Empty --
   --------------

   function Is_Empty (Container : Multiset) return Boolean is
      B : constant Boolean := Is_Empty (Container.Map);

   begin
      Lemma_Empty (Container);
      return B;
   end Is_Empty;

   --------------------------------
   -- Lemma_Contains_Cardinality --
   --------------------------------

   procedure Lemma_Contains_Cardinality
     (Container : Multiset; Element : Element_Type) is
   begin
      Lemma_Remove (Container.Map, Element);
   end Lemma_Contains_Cardinality;

   -----------------
   -- Lemma_Empty --
   -----------------

   procedure Lemma_Empty (M : Multiset) is
   begin
      if not Is_Empty (M.Map) then
         declare
            Element : constant Element_Type := Choose (M.Map);
         begin
            pragma Assert (Contains (M, Element));
         end;
      end if;
   end Lemma_Empty;

   --------------
   -- Lemma_Eq --
   --------------

   procedure Lemma_Eq (M1, M2 : Maps.Map) is
   begin
      if Length (M1) /= 0 then
         declare
            E : constant Element_Type := Choose (M2);
         begin
            Lemma_Remove (M1, E);
            Lemma_Eq (Remove (M1, E), Remove (M2, E));
         end;
      end if;
   end Lemma_Eq;

   ---------------------------
   -- Lemma_Leq_Cardinality --
   ---------------------------

   procedure Lemma_Leq_Cardinality (Left, Right : Maps.Map) is
      E : Element_Type;
   begin
      if not Is_Empty (Left) then
         E := Choose (Left);

         Lemma_Remove (Left, E);

         Lemma_Remove (Right, E);
         Lemma_Leq_Cardinality (Remove (Left, E), Remove (Right, E));
      end if;
   end Lemma_Leq_Cardinality;

   -----------------------------------
   -- Lemma_Nb_Occurence_Equivalent --
   -----------------------------------

   procedure Lemma_Nb_Occurence_Equivalent
     (Container : Multiset; Element_1, Element_2 : Element_Type) is
   begin
      if not Has_Key (Container.Map, Element_1) then
         Lemma_Has_Key_Equivalent (Container.Map, Element_1);
         pragma
           Assert
             (Nb_Occurence (Container, Element_1)
              = Nb_Occurence (Container, Element_2));
      elsif not Has_Key (Container.Map, Element_2) then
         Lemma_Has_Key_Equivalent (Container.Map, Element_2);
         pragma
           Assert
             (Nb_Occurence (Container, Element_1)
              = Nb_Occurence (Container, Element_2));
      else
         Lemma_Get_Equivalent (Container.Map, Element_1, Element_2);
         pragma
           Assert
             (Nb_Occurence (Container, Element_1)
              = Nb_Occurence (Container, Element_2));
      end if;
   end Lemma_Nb_Occurence_Equivalent;

   ------------------
   -- Lemma_Remove --
   ------------------

   procedure Lemma_Remove (M : Maps.Map; K : Element_Type) is
      L : constant Element_Type := Choose (M);
   begin
      if not Equivalent_Elements (L, K) then
         Lemma_Remove (Remove (M, L), K);
         Lemma_Eq (Remove (Remove (M, L), K), Remove (Remove (M, K), L));
         Lemma_Remove (Remove (M, K), L);
      else
         Lemma_Eq (Remove (M, L), Remove (M, K));
      end if;
   end Lemma_Remove;

   ------------------
   -- Nb_Occurence --
   ------------------

   function Nb_Occurence
     (Container : Multiset; Element : Element_Type) return Big_Natural
   is (if Has_Key (Container.Map, Element)
       then Get (Container.Map, Element)
       else 0);

   ------------
   -- Remove --
   ------------

   function Remove
     (Container : Multiset; Element : Element_Type; Count : Big_Positive := 1)
      return Multiset is
   begin
      if Count = Nb_Occurence (Container, Element) then
         return Remove_All (Container, Element);
      else
         declare
            Result : Multiset;

         begin
            Result.Map :=
              Set
                (Container.Map,
                 Element,
                 Nb_Occurence (Container, Element) - Count);
            Lemma_Contains_Cardinality (Container, Element);
            Result.Card := Cardinality (Container) - Count;

            Lemma_Remove (Result.Map, Element);
            Lemma_Remove (Container.Map, Element);

            Lemma_Eq
              (Remove (Result.Map, Element), Remove (Container.Map, Element));

            return Result;
         end;
      end if;

   end Remove;

   ----------------
   -- Remove_All --
   ----------------

   function Remove_All
     (Container : Multiset; Element : Element_Type) return Multiset
   with
     Refined_Post =>
       (Static =>
          Invariant (Remove_All'Result.Map, Remove_All'Result.Card)
          and then not Contains (Remove_All'Result, Element)
          and then
            Cardinality (Remove_All'Result)
            = Cardinality (Container) - Nb_Occurence (Container, Element)
          and then Equal_Except (Container, Remove_All'Result, Element)
          and then
            Map_Logic_Equal
              (Remove_All'Result.Map, Remove (Container.Map, Element)))
   is
   begin
      return Set : Multiset do
         Set.Map := Remove (Container.Map, Element);
         Lemma_Remove (Container.Map, Element);
         Set.Card := Container.Card - Get (Container.Map, Element);
      end return;
   end Remove_All;

   -----------
   -- Union --
   -----------

   function Sum (Left : Multiset; Right : Multiset) return Multiset
   is separate;

end SPARK.Containers.Functional.Multisets;
