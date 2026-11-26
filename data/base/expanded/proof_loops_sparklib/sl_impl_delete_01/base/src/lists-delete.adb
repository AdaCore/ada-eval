--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
procedure Delete
  (Container : in out List; Position : in out Cursor; Count : Count_Type)
is
   N : Node_Array renames Container.Nodes;
   X : Count_Type;

begin
   if not Has_Element (Container => Container, Position => Position) then
      raise Constraint_Error with "Position cursor has no element";
   end if;

   if Position.Node = Container.First then
      Delete_First (Container, Count);
      Position := No_Element;
      return;
   end if;

   if Count = 0 then
      Position := No_Element;
      return;
   end if;

   for Index in 1 .. Count loop

      X := Position.Node;

      if X = Container.Last then
         Position := No_Element;

         --  First /= Last (Position is Last but not First), so at least two
         --  nodes remain and Unlink_Last_Node's Length >= 2 precondition
         --  holds.

         Lemma_Reachable_Def (Container.First, Memory (Container));

         Unlink_Last_Node (Container);
         Container.Length := Container.Length - 1;
         Free (Container, X);
         return;
      end if;

      --  Advance past the interior node X: Q = Next (X) is active (X is
      --  interior, so Next (X) /= 0) and, by acyclicity, distinct from X
      --  and First. Read Q before unlinking X; Unlink_Interior_Node and
      --  Free both keep the active set (minus X) and First, so Q stays a
      --  valid, non-First cursor across the two calls.

      Lemma_Reachable_Closed_By_Next (Container.First, Memory (Container));
      Position.Node := N (X).Next;

      Lemma_Reachable_Is_Acyclic (Container.First, X, Memory (Container));
      Lemma_Reachable_Included (Container.First, X, Memory (Container));
      Lemma_Reachable_Def (X, Memory (Container));

      Unlink_Interior_Node (Container, X);
      Container.Length := Container.Length - 1;
      Free (Container, X);
   end loop;

   Position := No_Element;
end Delete;
