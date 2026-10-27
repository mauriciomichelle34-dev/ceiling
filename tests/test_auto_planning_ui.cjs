const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'..','planner.js'),'utf8');
function actualFunction(name) {
  const start=source.search(new RegExp(`^(?:async )?function ${name}\\(`,'m'));
  assert.ok(start>=0,name);
  const tail=source.slice(start);
  const next=tail.slice(1).search(/^(?:async )?function /m);
  return next<0?tail:tail.slice(0,next+1);
}
function fixture() {
  const elements={diameter:{value:'10'},gap:{value:'1'},photoSelect:{append(){}},actions:{hidden:false},pointHint:{textContent:''}};
  const calls=[];
  const plan={count:8,points:[],plan_id:'saved',diameter_mm:10,gap_mm:1,sampling_order:{layer_count:2}};
  const context={photos:[],current:null,selected:null,busy:true,
    $:id=>elements[id],document:{createElement:()=>({})},
    loadImage:async url=>({url}),calibration:()=>({dish_diameter_mm:90}),
    numeric:id=>Number(elements[id].value),capture(){},notice(){},renderTable(){},controls(){},draw(){},
    showPhoto(photo){context.current=photo;calls.push('show');},
    request:async (route,body)=>{calls.push({route,body});return plan;}};
  vm.createContext(context);
  vm.runInContext(actualFunction('createPlanForCurrentPhoto'),context);
  vm.runInContext(actualFunction('addRecognizedPhoto'),context);
  return {context,calls,plan};
}
test('upload and camera shared entry automatically plans even while upload is busy, without PLC writes',async()=>{
  const {context,calls,plan}=fixture();
  await context.addRecognizedPhoto({id:'photo',original:'/original',region:'/region',region_outline:'/outline'});
  assert.equal(context.photos.length,1);
  assert.equal(context.current.plan,plan);
  assert.equal(calls[0],'show');
  const requests=calls.filter(c=>typeof c==='object');
  assert.equal(requests.length,1);
  assert.equal(requests[0].route,'/api/plans/create');
  assert.equal(requests[0].body.diameter_mm,10);
  assert.equal(requests[0].body.gap_mm,1);
  assert.equal('default_depth_mm' in requests[0].body,false);
});
test('planning failure leaves recognized photo available for manual correction and never sends PLC',async()=>{
  const {context,calls}=fixture();
  context.request=async(route)=>{calls.push(route);throw new Error('planning failed');};
  await assert.rejects(context.addRecognizedPhoto({id:'photo',original:'o',region:'r',region_outline:'e'}),/planning failed/);
  assert.equal(context.photos.length,1);
  assert.equal(context.current.plan,null);
  assert.equal(calls.at(-1),'/api/plans/create');
});
test('multiple images receive their own plans once',async()=>{
  const {context,calls}=fixture();
  for(const id of ['one','two']) await context.addRecognizedPhoto({id,original:'o',region:'r',region_outline:'e'});
  const requests=calls.filter(c=>typeof c==='object');
  assert.deepEqual(requests.map(r=>r.body.prediction_id),['one','two']);
  assert.ok(context.photos.every(p=>p.plan));
});
