const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'..','automatic_ui.js'),'utf8');
function actualFunction(name) {
  const start=source.search(new RegExp(`^(?:async )?function ${name}\\(`,'m'));
  assert.ok(start>=0,name);
  const tail=source.slice(start);
  const next=tail.slice(1).search(/^(?:async )?function /m);
  return next<0?tail:tail.slice(0,next+1);
}
function fixture() {
  const elements={};
  for(const id of ['automaticSaveStart','automaticStartSaved','automaticStop','automaticMode',
    'automaticSource','automaticCycleTimeout','automaticStartHint','diameter','gap']) elements[id]={value:'',disabled:false,textContent:''};
  elements.automaticSource.value='plc_camera'; elements.automaticMode.value='execute';
  elements.automaticCycleTimeout.value='1800';elements.diameter.value='10';elements.gap.value='1';
  const meta={width:5472,height:3648,calibration:{dish_diameter_mm:90,mm_per_pixel:.04}};
  const origin={kind:'fixed_screw',x_px:4045,y_px:3010,image_width:5472,image_height:3648};
  const calls=[];
  const context={current:null,ready:true,busy:false,automaticEnabled:false,
    automaticSnapshot:{configured:false},cameraSnapshot:{last_capture:{processing:{kind:'analysis',meta}}},
    $:id=>elements[id],calibration:()=>{throw Error('No current photo');},
    coordinateOrigin(m){if(m.width!==5472||m.height!==3648)throw Error('照片尺寸不匹配');return origin;},
    numeric(id){const value=Number(elements[id].value);if(!Number.isFinite(value)||value<0||(id==='diameter'&&value===0))throw Error('参数无效');return value;},
    fetch:async()=>({json:async()=>({plc_camera_trigger_available:true})}),
    automaticRequest:async(action,body)=>{calls.push({action,body});return {enabled:true};},
    renderAutomatic(){},notice(message,error){calls.push({message,error});},refreshAutomatic(){}};
  vm.createContext(context);
  for(const name of ['automaticReference','updateAutomaticControls','startAutomatic'])vm.runInContext(actualFunction(name),context);
  return {context,elements,meta,origin,calls};
}
test('refresh reuses successful camera calibration and enables start without sending any request',()=>{
  const {context,elements,calls}=fixture();
  context.updateAutomaticControls();
  assert.equal(elements.automaticSaveStart.disabled,false);
  assert.match(elements.automaticStartHint.textContent,/复用最近一次/);
  assert.equal(calls.length,0);
});
test('explicit start uses the same reference that enabled the button',async()=>{
  const {context,origin,calls}=fixture();
  await context.startAutomatic(false);
  const request=calls.find(c=>c.action==='start');
  assert.ok(request);
  assert.equal(request.body.settings.image_width,5472);
  assert.equal(request.body.settings.image_height,3648);
  assert.equal(request.body.settings.calibration.coordinate_origin,origin);
  assert.equal(request.body.settings.input_source,'plc_camera');
  assert.equal(request.body.settings.diameter_mm,10);
  assert.equal(request.body.settings.gap_mm,1);
});
test('failed, unprocessed, or automatic camera records never become a fallback reference',async()=>{
  for(const frame of [{processing_error:'failed'}, {}, {processing:{kind:'automatic'}},
    {processing_error:'failed',processing:fixture().context.cameraSnapshot.last_capture.processing}]) {
    const {context,elements,calls}=fixture();context.cameraSnapshot.last_capture=frame;
    context.updateAutomaticControls();
    assert.equal(elements.automaticSaveStart.disabled,true);
    assert.match(elements.automaticStartHint.textContent,/参考照片/);
    await context.startAutomatic(false);
    assert.equal(calls.some(c=>c.action==='start'),false);
  }
});
test('invalid size and old dish calibration keep start disabled with a reason',()=>{
  for(const change of [meta=>meta.width=100,meta=>meta.calibration.dish_diameter_mm=100]) {
    const {context,elements,meta}=fixture();change(meta);context.updateAutomaticControls();
    assert.equal(elements.automaticSaveStart.disabled,true);
    assert.ok(elements.automaticStartHint.textContent.length);
  }
});
test('current photo and its validation take precedence over a stale camera reference',()=>{
  const {context,elements,meta}=fixture();
  context.current={meta:{...meta,width:2000}};
  context.calibration=()=>{throw Error('当前照片尺寸无效');};
  context.updateAutomaticControls();
  assert.equal(elements.automaticSaveStart.disabled,true);
  assert.equal(elements.automaticStartHint.textContent,'当前照片尺寸无效');
  context.calibration=()=>meta.calibration;
  assert.equal(context.automaticReference().reused,false);
});
test('health loading, busy processing, running workflow and active tasks keep start locked',()=>{
  for(const overrides of [{ready:false},{busy:true},{automaticEnabled:true},
    {automaticSnapshot:{active_job_id:'job'}},{automaticSnapshot:{camera_busy:true}}]) {
    const {context,elements}=fixture();Object.assign(context,overrides);context.updateAutomaticControls();
    assert.equal(elements.automaticSaveStart.disabled,true);
    assert.ok(elements.automaticStartHint.textContent.length);
  }
});
test('a disconnected recognition service prevents even an explicit start request',async()=>{
  const {context,calls}=fixture();context.ready=false;
  await context.startAutomatic(false);
  assert.equal(calls.some(c=>c.action==='start'),false);
});
