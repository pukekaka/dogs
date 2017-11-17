from idautils import *
from idaapi import *
from idc import *
ea = BeginEA()
for funcea in Functions(SegStart(ea), SegEnd(ea)):
    functionName = GetFunctionName(funcea)
    for (startea, endea) in Chunks(funcea):
        for head in Heads(startea, endea):
            print functionName, ":", "0x%08x"%(head), ":", GetDisasm(head)