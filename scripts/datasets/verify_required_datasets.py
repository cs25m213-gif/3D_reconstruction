"""Read-only dataset audit: sizes, ZIP directory, sampled CRC, extracted files and partials.

This is bounded integrity sampling, not a full re-hash/CRC of every archive and
not a camera/LiDAR alignment certificate. No downloads, deletion or GPU use.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import zipfile


def audit(root, manifest):
    rows=[]
    source=manifest.read_text().split('\n#__MANIFEST__\n',1)[1]
    for line in source.splitlines():
        if not line.startswith('required\t'):continue
        _,group,name,size,sha,url=line.split('\t')
        wanted=int(size)
        possible=[root/'vnext_downloads'/group/name,root/group/name,root/Path(name).name]
        exact=next((p for p in possible if p.is_file() and p.stat().st_size==wanted),None)
        found=exact or next((p for p in possible if p.is_file()),None)
        row={'key':group+'/'+name,'expected_bytes':wanted,'path':str(found) if found else None,
             'actual_bytes':found.stat().st_size if found else None,'integrity':'missing',
             'extraction':'not_applicable','sample_crc_members':0}
        if found:
            row['integrity']='size_match' if exact else 'incomplete_size'
        partial=root/'vnext_downloads'/group/(name+'.part')
        if partial.exists():row['partial_bytes']=partial.stat().st_size
        if exact and name.endswith('.zip'):
            output=root/'vnext_downloads/extracted'/group/str(Path(name).with_suffix(''))
            marker=output/'.geostreamfusion_complete.json'
            try:
                marked=json.loads(marker.read_text()) if marker.exists() else {}
                with zipfile.ZipFile(exact) as archive:
                    members=[i for i in archive.infolist() if not i.is_dir()]
                    counts=Counter(Path(i.filename).suffix.lower() for i in members)
                    row['archive_members']=len(members);row['types']=dict(counts)
                    row['missing_extracted_members']=sum(not (output/i.filename).is_file() for i in members)
                    eligible=[i for i in members if 0<i.file_size<=8*1024**2]
                    samples=[eligible[i] for i in sorted({0,len(eligible)//2,len(eligible)-1})] if eligible else []
                    for item in samples:
                        # Reading to EOF asks zipfile to validate this member's CRC.
                        with archive.open(item) as stream:
                            for block in iter(lambda:stream.read(1024**2),b''):pass
                        if (output/item.filename).stat().st_size!=item.file_size:
                            raise ValueError('Extracted size differs: '+item.filename)
                    row['sample_crc_members']=len(samples)
                    row['integrity']='zip_directory_and_sample_crc_pass' if samples else 'zip_directory_only'
                    row['extraction']='complete_inventory' if (marked.get('archive_size')==wanted and not row['missing_extracted_members']) else 'incomplete_or_unmarked'
            except (OSError,ValueError,zipfile.BadZipFile,RuntimeError) as exc:
                row['integrity']='failed';row['error']=str(exc)
        elif exact and name.endswith('.bag'):
            with exact.open('rb') as stream:
                row['integrity']='bag_header_and_size_pass' if stream.read(13)==b'#ROSBAG V2.0\n' else 'invalid_bag_header'
        rows.append(row)
    return {'created_unix':time.time(),'scope':'read-only inventory and bounded ZIP CRC sampling; not full-file checksum or geometric validation',
            'files':rows,'counts':dict(Counter(r['integrity'] for r in rows)),
            'uavscenes_usable_storage':all(r['integrity']=='zip_directory_and_sample_crc_pass' and r['extraction']=='complete_inventory' for r in rows if r['key'].startswith('uavscenes/'))}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root,a.manifest)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
