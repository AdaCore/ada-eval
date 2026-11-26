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

separate (JSON.Tokenizers)
procedure Scan_Literal
  (Stream     : in out Streams.Stream;
   First      : Character;
   Next_Token : out Token;
   Status     : out Token_Status)
is
   Buffer : String (1 .. 5) := (others => ' ');
   Last   : Natural range 0 .. 5;

   C   : Character;
   EOF : Boolean;
begin
   Next_Token := (Kind => Invalid_Token, others => <>);
   Status := Success;

   Buffer (1) := First;
   Last := 1;

   loop
      pragma Loop_Invariant (not Streams.Has_Buffered_Character (Stream));
      pragma Loop_Invariant
        (Streams.Length (Stream) = Streams.Length (Stream)'Loop_Entry);
      pragma Loop_Invariant (Last in 1 .. 5);
      pragma Loop_Invariant
        (Streams.Position (Stream) >= Streams.Position (Stream)'Loop_Entry);
      pragma Loop_Variant (Increases => Streams.Position (Stream));

      Streams.Read_Character (Stream, C, EOF);
      exit when EOF;
      exit when C not in 'a' .. 'z' or else Last = 5;
      Last := Last + 1;
      Buffer (Last) := C;
   end loop;

   if not EOF then
      Streams.Write_Character (Stream, C);
   end if;

   if Buffer (1 .. Last) = "true" then
      Next_Token :=
        (Kind => Boolean_Token, Boolean_Value => True, others => <>);
   elsif Buffer (1 .. Last) = "false" then
      Next_Token :=
        (Kind => Boolean_Token, Boolean_Value => False, others => <>);
   elsif Buffer (1 .. Last) = "null" then
      Next_Token := (Kind => Null_Token, others => <>);
   else
      Status := Error_Invalid_Literal;
   end if;
end Scan_Literal;
