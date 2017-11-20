# -*- coding: utf-8 -*-

from idautils import *
from idaapi import *
from idc import *
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


idc.Wait()
ea = BeginEA()
output_directory = 'E:/Works/Data/samples/output/'
filename = idc.AskFile(1, "*.*", "Save list of basic blocks")
basename = idc.GetInputFile()
filename = basename + ".json"
fp = open(output_directory + filename, 'w')
count = 0
line_list = []
for funcea in Functions(SegStart(ea), SegEnd(ea)):
    functionName = GetFunctionName(funcea)
    for (startea, endea) in Chunks(funcea):
        for head in Heads(startea, endea):
            alld = functionName + ":" + "0x%08x" % (head) + ":" + GetDisasm(head) + "\n"

            disasm = GetDisasm(head)
            disasm = unicode(disasm, errors='ignore').encode('utf-8').strip()
            fname = functionName
            fname = unicode(fname, errors='ignore').encode('utf-8').strip()
            addr = "0x%08x" % (head)
            addr = unicode(addr, errors='ignore').encode('utf-8').strip()

            opcode = GetMnem(head)
            opcode = unicode(opcode, errors='ignore').encode('utf-8').strip()

            opnd1 = GetOpnd(head, 0)
            opnd1 = unicode(opnd1, errors='ignore').encode('utf-8').strip()

            opnd2 = GetOpnd(head, 1)
            opnd2 = unicode(opnd2, errors='ignore').encode('utf-8').strip()

            opnd3 = GetOpnd(head, 2)
            opnd3 = unicode(opnd3, errors='ignore').encode('utf-8').strip()

            optype1 = GetOpType(head, 0)
            # optype1 = unicode(optype1, errors='ignore').encode('utf-8').strip()

            optype2 = GetOpType(head, 1)
            # optype2 = unicode(optype2, errors='ignore').encode('utf-8').strip()

            optype3 = GetOpType(head, 2)
            # optype3 = unicode(optype3, errors='ignore').encode('utf-8').strip()

            opndv1 = GetOperandValue(head, 0)
            # opndv1 = unicode(opndv1, errors='ignore').encode('utf-8').strip()

            opndv2 = GetOperandValue(head, 1)
            # opndv2 = unicode(opndv2, errors='ignore').encode('utf-8').strip()

            opndv3 = GetOperandValue(head, 2)
            # opndv3 = unicode(opndv3, errors='ignore').encode('utf-8').strip()

            split_disasm = disasm.split(';')
            split_disasm_size = len(split_disasm)

            if split_disasm_size > 1:
                comment = split_disasm[split_disasm_size - 1]
                comment = unicode(comment, errors='ignore').encode('utf-8').strip()
            else:
                comment = "" + "\n"
                comment = unicode(comment, errors='ignore').encode('utf-8').strip()

            inst = {
                "disasm": disasm,
                "fname": fname,
                "addr": addr,
                "opcode": opcode,
                "opnd1": opnd1,
                "opnd2": opnd2,
                "opnd3": opnd3,
                "optype1": optype1,
                "optype2": optype2,
                "optype3": optype3,
                "opndv1": opndv1,
                "opndv2": opndv2,
                "opndv3": opndv3,
                "comment": comment,
                "line": count
                }

            # jstr = json.dumps(inst)
            # fp.write(jstr)
            line_list.append(inst)
            count += 1

contents = {"filename": basename, "insts": line_list}
jsonString = json.dumps(contents)
fp.write(jsonString)

fp.close()
idc.Exit(0)


# GetOPType
# o_void  =      0  # No Operand
# o_reg  =       1  # General Register (al,ax,es,ds...)    reg
# o_mem  =       2  # Direct Memory Reference  (DATA)      addr
# o_phrase  =    3  # Memory Ref [Base Reg + Index Reg]    phrase
# o_displ  =     4  # Memory Reg [Base Reg + Index Reg + Displacement] phrase+addr
# o_imm  =       5  # Immediate Value                      value
# o_far  =       6  # Immediate Far Address  (CODE)        addr
# o_near  =      7  # Immediate Near Address (CODE)        addr
# 8: I don't remember what... something related to MOVs

# GetOperandValue
# operand is an immediate value  => immediate value
# operand has a displacement     => displacement
# operand is a direct memory ref => memory address
# operand is a register          => register number
# operand is a register phrase   => phrase number
# otherwise                      => -1


# GetOPType
# o_void  =      0  # No Operand
# o_reg  =       1  # General Register (al,ax,es,ds...)    reg
# o_mem  =       2  # Direct Memory Reference  (DATA)      addr
# o_phrase  =    3  # Memory Ref [Base Reg + Index Reg]    phrase
# o_displ  =     4  # Memory Reg [Base Reg + Index Reg + Displacement] phrase+addr
# o_imm  =       5  # Immediate Value                      value
# o_far  =       6  # Immediate Far Address  (CODE)        addr
# o_near  =      7  # Immediate Near Address (CODE)        addr
# 8: I don't remember what... something related to MOVs

# GetOperandValue
# operand is an immediate value  => immediate value
# operand has a displacement     => displacement
# operand is a direct memory ref => memory address
# operand is a register          => register number
# operand is a register phrase   => phrase number
# otherwise                      => -1