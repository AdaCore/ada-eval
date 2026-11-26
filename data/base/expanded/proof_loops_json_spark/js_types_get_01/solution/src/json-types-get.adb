--  SPDX-License-Identifier: Apache-2.0
--
--  Copyright (c) 2016 onox <denkpadje@gmail.com>
--
--  Licensed under the Apache License, Version 2.0 (the "License");
--  you may not use this file except in compliance with the License.
--  You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
--  Unless required by applicable law or agreed to in writing, software
--  distributed under the License is distributed on an "AS IS" BASIS,
--  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
--  See the License for the specific language governing permissions and
--  limitations under the License.

separate (JSON.Types)
function Get
  (Object : not null access constant JSON_Value; Index : Positive)
   return access constant JSON_Value
is
   Node  : access constant JSON_Value := Object.First_Child;
   Count : Positive := 1;
begin
   while Node /= null loop
      pragma Loop_Invariant (Count <= Index);
      pragma Loop_Variant (Structural => Node);

      if Count = Index then
         return Node;
      end if;
      Count := Count + 1;
      Node := Node.Next;
   end loop;
   return null;
end Get;
