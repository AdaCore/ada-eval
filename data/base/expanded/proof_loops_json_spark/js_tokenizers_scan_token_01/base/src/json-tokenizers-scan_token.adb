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
procedure Scan_Token
  (Stream     : in out Streams.Stream;
   Next_Token : out Token;
   Status     : out Token_Status;
   Expect_EOF : Boolean := False)
is
   use Ada.Characters.Latin_1;

   C   : Character;
   EOF : Boolean;
begin
   Next_Token := (Kind => Invalid_Token, others => <>);
   Status := Success;

   loop
      pragma
        Loop_Variant
          (Decreases =>
             Streams.Length (Stream)
             - Streams.Position (Stream)
             + 1
             + (if Streams.Has_Buffered_Character (Stream) then 1 else 0));

      --  Read the next character and decide which token it could be
      --  part of, skipping whitespace
      Streams.Read_Character (Stream, C, EOF);
      exit when EOF or else C not in ' ' | HT | LF | CR;
   end loop;

   if EOF then
      if Expect_EOF then
         Next_Token := (Kind => EOF_Token, others => <>);
      else
         Status := Error_Unexpected_EOF;
      end if;
      return;
   end if;

   if Expect_EOF then
      Status := Error_Expected_EOF;
      return;
   end if;

   case C is
      when '['                    =>
         Next_Token := (Kind => Begin_Array_Token, others => <>);

      when '{'                    =>
         Next_Token := (Kind => Begin_Object_Token, others => <>);

      when ']'                    =>
         Next_Token := (Kind => End_Array_Token, others => <>);

      when '}'                    =>
         Next_Token := (Kind => End_Object_Token, others => <>);

      when ':'                    =>
         Next_Token := (Kind => Name_Separator_Token, others => <>);

      when ','                    =>
         Next_Token := (Kind => Value_Separator_Token, others => <>);

      when '"'                    =>
         Scan_String (Stream, Next_Token, Status);

      when '0' .. '9' | '+' | '-' =>
         Scan_Number (Stream, C, Next_Token, Status);

      when 'a' .. 'z'             =>
         Scan_Literal (Stream, C, Next_Token, Status);

      when others                 =>
         Status := Error_Unexpected_Character;
   end case;
end Scan_Token;
