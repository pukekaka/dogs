from idautils import *
from idaapi import *
from idc import *
import json


idc.Wait()
output_directory = 'E:/Works/Data/samples/output/'
filename = idc.AskFile(1, "*.*", "Save list of basic blocks")
basename = idc.GetInputFile()
filename = basename + ".bblist"
fp = open(output_directory + filename, 'w')
# ea = BeginEA()
# func = Functions(SegStart(ea), SegEnd(ea))
funcs = Functions()
bblist = []
count = 0
for func in funcs:
	flowchart = FlowChart(get_func(func))
	for block in flowchart:
		startEA = block.startEA
		endEA = block.endEA
		curName = GetFunctionName(startEA);
		dem = idc.Demangle(curName, idc.GetLongPrm(INF_SHORT_DN));
		if dem != None:
			curName = dem;
#
		first = startEA
		h = Heads(startEA, endEA)
		for i in h:
			mnem = idc.GetMnem(i)
			if mnem == "call" and i != endEA:
				# print >> fp, "%#010x %#010x %s" % (first, idc.NextHead(i, endEA + 1) - 1, curName)
				start = "%#010x" % (first)
				end = "%#010x" % (idc.NextHead(i, endEA + 1) - 1)
				fname = "%s" % (curName)
				# contents = str(count) + ", " +start + ":" + end + ":" + fname + "\n"
				# fp.write(contents)
				inst = {
					"start": start,
					"end": end,
					"fname": fname,
					"line": count
				}
				bblist.append(inst)
				first = idc.NextHead(i, endEA + 1)
				count += 1

		if first < endEA:
			start = "%#010x" % (first)
			end = "%#010x" % (endEA - 1)
			fname = "%s" % (curName)
			# print >> fp, "%#010x %#010x %s" % (first, endEA - 1, curName)
			# contents = str(count) + "===============" + start + ":" + end + ":" + fname + "\n"
			# fp.write(contents)
			inst={
				"start": start,
				"end": end,
				"fname": fname,
				"line": count
			}
			bblist.append(inst)
			count += 1

contents = {"filename": basename, "bblist": bblist}
jsonString = json.dumps(contents)
fp.write(jsonString)

fp.close()
idc.Exit(0)