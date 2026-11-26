--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
function Find
  (Container : List; Item : Element_Type; Position : Cursor := No_Element)
   return Cursor
is
   From : Count_Type := Position.Node;

begin
   if From = 0 then
      From := Container.First;
   end if;

   --  The active list is closed under Next, so once From is reachable from
   --  First its successor is reachable (or 0).

   Lemma_Reachable_Closed_By_Next (Container.First, Memory (Container));

   while From /= 0 loop
      pragma
        Loop_Variant
          (Static =>
             (Decreases =>
                Memory_Index_Sets.Length
                  (Reachable_Set (From, Memory (Container)))));

      Lemma_Reachable_Is_Acyclic
        (Container.First, From, Memory (Container));
      Lemma_Reachable_Def (From, Memory (Container));

      if Container.Nodes (From).Element = Item then
         return (Node => From);
      end if;

      From := Container.Nodes (From).Next;
   end loop;

   return No_Element;
end Find;
