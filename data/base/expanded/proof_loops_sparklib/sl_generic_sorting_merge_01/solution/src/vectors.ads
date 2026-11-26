--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

--  A local model of the bounded-vector representation that
--  SPARK.Containers.Formal.Vectors declares in its private part and that
--  SPARK.Containers.Formal.Vectors.Impl operates on.  Only what
--  Generic_Sorting.Merge touches is kept; the representation itself
--  (Relaxed_Element,
--  Elements_Array, To_Array_Index, the Vector record and its Ghost_Predicate)
--  is copied verbatim from upstream.

pragma Ada_2022;

with SPARK.Containers.Types; use SPARK.Containers.Types;
with Element_Type_Defs;      use Element_Type_Defs;
with Index_Types;

package Vectors with SPARK_Mode, Always_Terminates is

   subtype Index_Type is Index_Types.Big_Index;
   use type Index_Types.Big_Base;
   --  Upstream Index_Type is a generic formal signed type, whose operators are
   --  directly visible; here it is a named subtype, so the operators of its
   --  base type have to be made visible explicitly.

   subtype Extended_Index is
     Index_Type'Base
       range Index_Type'Pred (Index_Type'First) .. Index_Type'Last;

   No_Index : constant Extended_Index := Extended_Index'First;

   Last_Count : constant Count_Type :=
     (if Index_Type'Last < Index_Type'First
      then 0
      elsif (if Index_Type'First > 0
             then
               Index_Type'Pos (Index_Type'Last)
               - Index_Type'Pos (Index_Type'First)
               < Count_Type'Pos (Count_Type'Last)
             else
               Index_Type'Pos (Index_Type'Last)
               < Count_Type'Pos (Count_Type'Last)
                 + Index_Type'Pos (Index_Type'First))
      then
        Count_Type'Val
          (Index_Type'Pos (Index_Type'Last)
           - Index_Type'Pos (Index_Type'First)
           + 1)
      else Count_Type'Last);
   --  Maximal capacity of any vector. It is the minimum of the size of the
   --  index range and the last possible Count_Type.

   subtype Capacity_Range is Count_Type range 0 .. Last_Count;

   type Vector (Capacity : Capacity_Range) is private;

   --  The following ghost helpers express, without any model, the exact
   --  conditions under which the insertion operations raise en exception. They
   --  are used in the Exit_Cases below so that those contracts state precisely
   --  when, and which, exception is raised.

   function Valid_Index_For_Insertion
     (Last : Extended_Index; Before : Extended_Index) return Boolean
   is ((Before in Index_Type'First .. Last)
       or else (Before /= No_Index and then Before - 1 = Last))
   with Ghost => Static, Global => null;
   --  Before designates a valid insertion point: an existing index, or the
   --  slot immediately after the last one (so that appending is allowed). A
   --  Before outside this set makes the insertion raise Constraint_Error.

   function Exceeds_Last_Count
     (Length : Count_Type; Count : Count_Type) return Boolean
   is (Count > 0 and then Length > Last_Count - Count)
   with Ghost => Static, Global => null;
   --  Inserting Count items would push the length past the maximum vector
   --  length Last_Count, which raises Constraint_Error.

   function Exceeds_Capacity
     (Length : Count_Type; Capacity : Capacity_Range; Count : Count_Type)
      return Boolean
   is (Count > 0
       and then Length <= Last_Count - Count
       and then Length > Capacity - Count)
   with Ghost => Static, Global => null;
   --  Inserting Count items fits within Last_Count but exceeds the allocated
   --  capacity, which raises Capacity_Error.

   function Last_Index (Container : Vector) return Extended_Index
   with Global => null;

   function Length (Container : Vector) return Capacity_Range
   with Global => null;

   function Capacity (Container : Vector) return Capacity_Range
   with Global => null;

   procedure Merge (Target : in out Vector; Source : in out Vector)
   with
     Global => null,
     Pre    =>
       --  The four guards upstream's Merge checks before it reaches the loop.
       --  The first three are its early returns and its explicit raise; the
       --  last two are the "others => Normal_Return" case of its own
       --  Exit_Cases, i.e. exactly the condition under which the Insert_Space
       --  call below returns rather than raising.
       Length (Source) /= 0
       and then Length (Target) /= 0
       and then Last_Index (Target) < Index_Type'Last
       and then not Exceeds_Last_Count (Length (Target), Length (Source))
       and then not Exceeds_Capacity
                      (Length (Target), Capacity (Target), Length (Source));

private

   --  The payload is wrapped in a one-field record marked
   --  Relaxed_Initialization, so that only the used prefix Elements
   --  (1 .. Last) needs to be initialized; the unused cells hold uninitialized
   --  Element_Type values. The V field is aliased so that Reference and
   --  Constant_Reference can hand out an access designating it.

   type Relaxed_Element is record
      V : aliased Element_Type;
   end record
   with Relaxed_Initialization;

   subtype Array_Index is Capacity_Range range 1 .. Capacity_Range'Last;
   type Elements_Array is array (Array_Index range <>) of Relaxed_Element;

   function "=" (L, R : Elements_Array) return Boolean is abstract;

   function To_Array_Index (Index : Index_Type'Base) return Count_Type'Base
   is
      --  We know that
      --    Index >= Index_Type'First
      --  hence we also know that
      --    Index - Index_Type'First >= 0

      --  The issue is that even though 0 is guaranteed to be a value in
      --  the type Index_Type'Base, there's no guarantee that the difference
      --  is a value in that type. To prevent overflow we use the wider
      --  of Count_Type'Base and Index_Type'Base to perform intermediate
      --  calculations.

      --  The array index subtype for all container element arrays always
      --  starts with 1.

      (if Index_Type'Base'Last >= Count_Type'Pos (Count_Type'Last)
       then Count_Type'Base (Index - No_Index)
       else Count_Type'Base (Index) - Count_Type'Base (No_Index))
   with
     Pre =>
       Index >= No_Index
       and then Index <= No_Index + Count_Type'Pos (Last_Count);

   type Vector (Capacity : Capacity_Range) is record
      Last     : Extended_Index := No_Index;
      Elements : Elements_Array (1 .. Capacity);
   end record
   with
     Ghost_Predicate =>
       (Static =>
          Last <= No_Index + Count_Type'Pos (Last_Count)
          and then To_Array_Index (Last) <= Capacity
          and then
            (for all I in 1 .. To_Array_Index (Last) =>
               Elements (I).V'Initialized));

   function Last_Index (Container : Vector) return Extended_Index
   is (Container.Last);

   function Length (Container : Vector) return Capacity_Range
   is (To_Array_Index (Container.Last));

   function Capacity (Container : Vector) return Capacity_Range
   is (Container.Capacity);

end Vectors;
