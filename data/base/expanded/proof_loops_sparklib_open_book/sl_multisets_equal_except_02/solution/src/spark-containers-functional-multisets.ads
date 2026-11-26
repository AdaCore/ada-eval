--
--  Copyright (C) 2022-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Warnings (Off, "unused variable ""Element""");
pragma Ada_2022;

with SPARK.Containers.Parameter_Checks;
with SPARK.Big_Integers; use SPARK.Big_Integers;

private with SPARK.Containers.Functional.Maps;

generic
   type Element_Type is private;

   with
     function Equivalent_Elements
       (Left : Element_Type; Right : Element_Type) return Boolean is "=";

   Enable_Handling_Of_Equivalence : Boolean := True;
   --  This constant should only be set to False when no particular handling
   --  of equivalence over keys is needed, that is, Equivalent_Keys defines a
   --  key uniquely.

   --  Ghost lemmas used to prove that Equivalent_Elements is an equivalence
   --  relation.

   with
     procedure Equivalent_Elements_Reflexive (X : Element_Type) is null
     with Ghost => Static;
   with
     procedure Equivalent_Elements_Symmetric (X, Y : Element_Type) is null
     with Ghost => Static;
   with
     procedure Equivalent_Elements_Transitive (X, Y, Z : Element_Type) is null
     with Ghost => Static;

package SPARK.Containers.Functional.Multisets with
    SPARK_Mode => On,
    Always_Terminates
is
   pragma
     Annotate
       (GNATcheck,
        Exempt_On,
        "Restrictions:No_Specification_Of_Aspect => Iterable",
        "The following usage of aspect Iterable has been reviewed"
        & "for compliance with GNATprove assumption"
        & " [SPARK_ITERABLE]");
   type Multiset is private
   with
     Default_Initial_Condition => (SPARKlib_Full => Is_Empty (Multiset)),
     Iterable                  =>
       (First       => Iter_First,
        Next        => Iter_Next,
        Has_Element => Iter_Has_Element,
        Element     => Iter_Element),
     Aggregate                 =>
       (Empty => Empty_Multiset, Add_Named => Aggr_Include),
     Annotate                  =>
       (GNATprove, Container_Aggregates, "Predefined_Maps");
   pragma
     Annotate
       (GNATcheck,
        Exempt_Off,
        "Restrictions:No_Specification_Of_Aspect => Iterable");
   --  Multisets are empty when default initialized.
   --  "For in" quantification over multisets should not be used.
   --  "For of" quantification over multisets iterates over elements.
   --  Occurrences in a multiset are counted modulo equivalence, the whole
   --  equivalence class is included/excluded at once. As equivalence classes
   --  might be infinite, quantification over elements of a finite multiset
   --  could be infinite. Thus, quantified expressions cannot be executed and
   --  should only be used in disabled ghost code. This is enforced by using
   --  the SPARKlib_Full assertion level.

   ----------------------
   -- Basic Operations --
   ----------------------

   function Nb_Occurence
     (Container : Multiset; Element : Element_Type) return Big_Natural
   with
     --  Get the number of occurence of Element in the Container

     Global   => null,
     Annotate => (GNATprove, Container_Aggregates, "Get");

   procedure Lemma_Nb_Occurence_Equivalent
     (Container : Multiset; Element_1, Element_2 : Element_Type)
     --  Nb_Occurence returns the same result on all equivalent elements
   with
     Ghost    => SPARKlib_Full,
     Global   => null,
     Annotate => (GNATprove, Automatic_Instantiation),
     Pre      =>
       Enable_Handling_Of_Equivalence
       and then Equivalent_Elements (Element_1, Element_2),
     Post     =>
       Nb_Occurence (Container, Element_1)
       = Nb_Occurence (Container, Element_2);

   function Contains
     (Container : Multiset; Element : Element_Type) return Boolean
   with
     --  Returns True iff Element occurs at least one in Container

     Global   => null,
     Post     =>
       (SPARKlib_Full =>
          Contains'Result = (Nb_Occurence (Container, Element) > 0)),
     Annotate => (GNATprove, Inline_For_Proof),
     Annotate => (GNATprove, Iterable_For_Proof, "Contains");

   function Choose (Container : Multiset) return Element_Type
   with
     --  Returns an element of the Multiset

     Global => null,
     Pre    => (SPARKlib_Defensive => not Is_Empty (Container)),
     Post   => (SPARKlib_Full => Contains (Container, Choose'Result));

   function Cardinality (Container : Multiset) return Big_Natural
   with
     --  The Cardinality of a Multiset is the number of equivalence classes in
     --  the Multiset taking into account the number of occurences of these
     --  equivalent classes.

     Global => null,
     Post   =>
       (SPARKlib_Full =>
          (for all Element of Container =>
             Nb_Occurence (Container, Element) <= Cardinality'Result));

   ------------------------
   -- Property Functions --
   ------------------------

   function Is_Empty (Container : Multiset) return Boolean
   with
     Global => null,
     Post   =>
       (SPARKlib_Full =>
          Is_Empty'Result = (for all Element of Container => False)
          and then Is_Empty'Result = (Cardinality (Container) = 0));
   --  A Multiset is empty if it has no element. In other words, if the number
   --  of occurence of all the elements is 0.

   function Equal_Except
     (Left : Multiset; Right : Multiset; Element : Element_Type) return Boolean
   with
     --  Check if all the elements of Left except the equivalence class of
     --  Element have the same number of occurences in Left and in Right and
     --  conversely.

     Import,
     Global => null,
     Post   =>
       (SPARKlib_Full =>
          Equal_Except'Result
          = ((for all E of Left =>
                (if not Equivalent_Elements (E, Element)
                 then Nb_Occurence (Left, E) = Nb_Occurence (Right, E)))
             and then
               (for all E of Right =>
                  (if not Equivalent_Elements (E, Element)
                   then Nb_Occurence (Left, E) = Nb_Occurence (Right, E)))));

   function Equal_Except_Right
     (Left : Multiset; Right : Multiset; Element : Element_Type) return Boolean
   with

     Global => null,
     Pre    =>
       (SPARKlib_Full =>
          (for all E of Left =>
             (if not Equivalent_Elements (E, Element)
              then Nb_Occurence (Left, E) = Nb_Occurence (Right, E)))),
     Post   =>
       (SPARKlib_Full =>
          Equal_Except_Right'Result
          = (for all E of Right =>
               (if not Equivalent_Elements (E, Element)
                then Nb_Occurence (Left, E) = Nb_Occurence (Right, E))));
   ----------------------------
   -- Construction Functions --
   ----------------------------

   --  For better efficiency of both proofs and execution, avoid using
   --  construction functions in annotations and rather use property functions.

   function Empty_Multiset return Multiset
   with
     --  Returns an empty Multiset

     Global => null,
     Post   => (SPARKlib_Full => Is_Empty (Empty_Multiset'Result));

   function Add (Container : Multiset; Element : Element_Type) return Multiset
   with
     Global => null,
     Post   =>
       (SPARKlib_Full =>
          Contains (Add'Result, Element)
          and then Cardinality (Add'Result) = Cardinality (Container) + 1
          and then
            Nb_Occurence (Add'Result, Element)
            = Nb_Occurence (Container, Element) + 1
          and then Equal_Except (Container, Add'Result, Element));
   --  Returns Container with the number of occurences of the equivalence class
   --  of Element incremented by one.

   function Add
     (Container : Multiset; Element : Element_Type; Count : Big_Positive)
      return Multiset
   with
     Global => null,
     Post   =>
       (SPARKlib_Full =>
          Contains (Add'Result, Element)
          and then Cardinality (Add'Result) = Cardinality (Container) + Count
          and then
            Nb_Occurence (Add'Result, Element)
            = Nb_Occurence (Container, Element) + Count
          and then Equal_Except (Container, Add'Result, Element));
   --  Returns Container with the number of occurences of the equivalence class
   --  of Element incremented by Count.

   function Remove_All
     (Container : Multiset; Element : Element_Type) return Multiset
   with
     Global => null,
     Pre    => (SPARKlib_Defensive => Contains (Container, Element)),
     Post   =>
       (SPARKlib_Full =>
          not Contains (Remove_All'Result, Element)
          and then
            Cardinality (Remove_All'Result)
            = Cardinality (Container) - Nb_Occurence (Container, Element)
          and then Equal_Except (Container, Remove_All'Result, Element));
   --  Returns Container with no occurences of the equivalence class of Element

   function Remove
     (Container : Multiset; Element : Element_Type; Count : Big_Positive := 1)
      return Multiset
   with
     Global => null,
     Pre    =>
       (SPARKlib_Defensive => Count <= Nb_Occurence (Container, Element)),
     Post   =>
       (SPARKlib_Full =>
          Nb_Occurence (Remove'Result, Element)
          = Nb_Occurence (Container, Element) - Count
          and then
            Cardinality (Remove'Result) = Cardinality (Container) - Count
          and then Equal_Except (Container, Remove'Result, Element));
   --  Returns Container with the number of occurences of the equivalence class
   --  of Element decremented by Count.

   --------------------------
   -- Instantiation Checks --
   --------------------------

   --  Check that the actual parameters follow the appropriate assumptions

   function Copy_Element (Item : Element_Type) return Element_Type
   is (Item);
   --  Elements of containers are copied by numerous primitives in this
   --  package. This function causes GNATprove to verify that such a copy is
   --  valid (in particular, it does not break the ownership policy of SPARK,
   --  i.e. it does not contain pointers that could be used to alias mutable
   --  data).

   package Eq_Elements_Checks is new
     SPARK.Containers.Parameter_Checks.Equivalence_Checks
       (T                   => Element_Type,
        Eq                  => Equivalent_Elements,
        Param_Eq_Reflexive  => Equivalent_Elements_Reflexive,
        Param_Eq_Symmetric  => Equivalent_Elements_Symmetric,
        Param_Eq_Transitive => Equivalent_Elements_Transitive);
   --  Check that the actual parameter for Equivalent_Elements is an
   --  equivalence relation.

   --------------------------------------------------
   -- Iteration Primitives Used For Quantification --
   --------------------------------------------------

   type Private_Key is private;

   function Iter_First (Container : Multiset) return Private_Key
   with Ghost => SPARKlib_Full, Global => null;

   function Iter_Has_Element
     (Container : Multiset; Key : Private_Key) return Boolean
   with Ghost => SPARKlib_Full, Global => null;

   function Iter_Next
     (Container : Multiset; Key : Private_Key) return Private_Key
   with
     Ghost  => SPARKlib_Full,
     Global => null,
     Pre    => Iter_Has_Element (Container, Key);

   function Iter_Element
     (Container : Multiset; Key : Private_Key) return Element_Type
   with
     Ghost  => SPARKlib_Full,
     Global => null,
     Pre    => Iter_Has_Element (Container, Key);

   ------------------------------------------
   -- Additional Primitives For Aggregates --
   ------------------------------------------

   function Aggr_Eq_Elements (Left, Right : Element_Type) return Boolean
   is (Equivalent_Elements (Left, Right))
   with
     Global   => null,
     Ghost    => Static,
     Annotate => (GNATprove, Inline_For_Proof),
     Annotate => (GNATprove, Container_Aggregates, "Equivalent_Keys");

   function Default_Count return Big_Natural
   is (0)
   with
     Ghost    => Static,
     Annotate => (GNATprove, Inline_For_Proof),
     Annotate => (GNATprove, Container_Aggregates, "Default_Item");

   procedure Aggr_Include
     (Container : in out Multiset; Element : Element_Type; Count : Big_Natural)
   with
     Global => null,
     Pre    => (SPARKlib_Defensive => Nb_Occurence (Container, Element) = 0),
     Post   =>
       (SPARKlib_Full =>
          Nb_Occurence (Container, Element) = Count
          and then
            (for all E of Container =>
               (if not Equivalent_Elements (E, Element)
                then
                  Nb_Occurence (Container, E)
                  = Nb_Occurence (Container'Old, E)))
          and then
            (for all E of Container'Old =>
               (if not Equivalent_Elements (E, Element)
                then
                  Nb_Occurence (Container, E)
                  = Nb_Occurence (Container'Old, E))));

private
   package Maps is new
     SPARK.Containers.Functional.Maps
       (Key_Type                   => Element_Type,
        Element_Type               => Big_Positive,
        Equivalent_Keys            => Equivalent_Elements,
        Equivalent_Keys_Reflexive  => Eq_Elements_Checks.Eq_Reflexive,
        Equivalent_Keys_Symmetric  => Eq_Elements_Checks.Eq_Symmetric,
        Equivalent_Keys_Transitive => Eq_Elements_Checks.Eq_Transitive);
   use Maps;

   type Multiset is record
      Map  : Maps.Map;
      Card : Big_Natural := 0;
   end record
   with Type_Invariant => (SPARKlib_Full => Invariant (Map, Card));

   function Invariant (Container : Map; Card : Big_Natural) return Boolean
   with Ghost => SPARKlib_Full;

   --------------------------------------------------
   -- Iteration Primitives Used For Quantification --
   --------------------------------------------------

   type Private_Key is new Maps.Private_Key;

   function Iter_First (Container : Multiset) return Private_Key
   is (Iter_First (Container.Map));

   function Iter_Has_Element
     (Container : Multiset; Key : Private_Key) return Boolean
   is (Iter_Has_Element (Container.Map, Key));

   function Iter_Next
     (Container : Multiset; Key : Private_Key) return Private_Key
   is (Iter_Next (Container.Map, Key));

   function Iter_Element
     (Container : Multiset; Key : Private_Key) return Element_Type
   is (Iter_Element (Container.Map, Key));

end SPARK.Containers.Functional.Multisets;
