--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
function Copy (Source : List; Capacity : Count_Type := 0) return List is

   C : constant Count_Type := Count_Type'Max (Source.Capacity, Capacity);
   P : List (C);

begin

   --  Copy the touched region as a single slice assignment (not field by
   --  field): it copies the whole node records, so Source's untouched tail
   --  with its uninitialized fields is carried over without a read of an
   --  uninitialized scalar.

   P.Nodes (1 .. Source.Capacity) := Source.Nodes (1 .. Source.Capacity);

   P.Length := Source.Length;
   P.First := Source.First;
   P.Last := Source.Last;

   if Source.Free < 0 or else C = Source.Capacity then

      --  No extra capacity is touched, so P mirrors Source's free
      --  encoding directly.

      P.Free := Source.Free;

      --  Source's free chain (when explicit) is copied verbatim -- same
      --  links on its reachable nodes, all within Source.Capacity <= C --
      --  so it is preserved from Source into P.

      Lemma_Free_List_Preserved (Source.Free, Memory (Source), Memory (P));

   else

      --  Source.Free >= 0 and C > Source.Capacity: the extra cells
      --  Source.Capacity + 1 .. C must join the free list. Thread them into
      --  a chain ending at Source's free head, mark them deallocated
      --  (Prev = -1), and publish P.Free (= the tail head) only once every
      --  exposed cell is initialized.

      for N in Source.Capacity + 1 .. C loop
         P.Nodes (N).Prev := -1;
         P.Nodes (N).Next := (if N = C then Source.Free else N + 1);

      end loop;

      P.Free := Source.Capacity + 1;

      --  The extra cells Source.Capacity + 1 .. C were threaded into a
      --  chain ending at Source's free head; rebuild the merged free list
      --  from its head P.Free (= Source.Capacity + 1) via the local lemma.
      --  Source's free nodes lie in 1 .. Source.Capacity (copied verbatim,
      --  so their links and Prev = -1 carry over), which is what the lemma
      --  needs of the continuation chain. This is proof-only code, kept
      --  inside this (already-taken) branch -- whose guard is exactly the
      --  merge condition -- so no separate run-time test is evaluated.

      Lemma_Free_List_Preserved (Source.Free, Memory (Source), Memory (P));
      Rebuild_Free_List (P, Source.Capacity + 1, Source.Free);
   end if;

   --  The active list is copied verbatim -- same links on its reachable
   --  nodes, all within Source.Capacity <= C -- so it is preserved from
   --  Source into the larger P.

   Lemma_Is_Acyclic_Preserved (Source.First, Memory (Source), Memory (P));
   Lemma_Reachable_Preserved (Source.First, Memory (Source), Memory (P));

   return P;
end Copy;
