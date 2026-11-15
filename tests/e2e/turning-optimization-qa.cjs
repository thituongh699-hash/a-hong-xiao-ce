const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const root = process.cwd();
  const qa = path.join(root, 'reports/optimization-qa');
  const fixture = JSON.parse(fs.readFileSync(path.join(qa, 'api-fixtures.json'), 'utf8'));
  const pageHtml = fs.readFileSync(path.join(root, 'static/parameter-optimization.html'), 'utf8');
  const reportHtml = fs.readFileSync(path.join(qa, 'served-report.html'), 'utf8');
  const browser = await chromium.launch({ executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe', headless: true });
  const page = await browser.newPage({ viewport: {width:1366,height:1000} });
  const errors = []; page.on('pageerror', e => errors.push(String(e)));
  let submitted;
  await page.route('http://optimization.test/**', async route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/optimization') return route.fulfill({contentType:'text/html',body:pageHtml});
    if (url.pathname === '/api/optimization/defaults') return route.fulfill({json:fixture.defaults});
    if (url.pathname === '/api/optimization/jobs' && route.request().method() === 'POST') {
      submitted = route.request().postDataJSON();
      return route.fulfill({status:201,json:{job_id:'browser-contract-check',status:'running'}});
    }
    if (url.pathname === '/api/optimization/jobs') return route.fulfill({json:fixture.jobs});
    if (url.pathname.endsWith('/report')) return route.fulfill({contentType:'text/html',body:reportHtml});
    if (url.pathname.includes('/files/')) {
      const name = path.basename(url.pathname);
      const file = path.join(root,'reports/turning-point-optimization-2026-10-08',name);
      return route.fulfill({body:fs.readFileSync(file),headers:{'Content-Type':'text/csv','Content-Disposition':`attachment; filename="${name}"`}});
    }
    return route.fulfill({status:404,body:'unknown route'});
  });
  await page.goto('http://optimization.test/optimization');
  await page.getByText('已有实验结果').waitFor();
  if ((await page.locator('#intraday_thresholds').inputValue()) !== '3, 4, 4.5, 5') throw Error('Defaults did not load');
  await page.screenshot({path:path.join(qa,'优化模块桌面.png'),fullPage:true});
  await page.locator('#intraday_thresholds').fill('4, bad');
  await page.locator('#submit-button').click();
  if (!(await page.locator('#form-message').textContent()).includes('有效数字')) throw Error('Invalid input was not rejected');
  if (submitted) throw Error('Invalid input submitted');
  await page.locator('#intraday_thresholds').fill('3,4,4.5,5');
  await page.locator('#submit-button').click();
  await page.waitForFunction(() => document.querySelector('#form-message').textContent.includes('任务已启动'));
  if (JSON.stringify(submitted.intraday_thresholds) !== '[3,4,4.5,5]') throw Error('Fractional thresholds not passed');
  await page.setViewportSize({width:390,height:844});
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw Error('Mobile page overflows');
  await page.screenshot({path:path.join(qa,'优化模块手机.png'),fullPage:true});
  await page.setViewportSize({width:1366,height:1000});
  await page.goto('http://optimization.test/api/optimization/jobs/turning-point-2026-10-08/report');
  const before = await page.locator('#grid tbody tr:visible').count();
  if (before !== 780) throw Error(`Expected 780 rows, got ${before}`);
  await page.locator('#search').fill('4.5');
  const after = await page.locator('#grid tbody tr:visible').count();
  if (!(after>0 && after<before)) throw Error('Grid filtering failed');
  const download = page.waitForEvent('download');
  await page.getByRole('link',{name:'全部训练参数表',exact:true}).click();
  const downloaded = await download;
  if (downloaded.suggestedFilename() !== 'training-grid.csv') throw Error('Wrong download');
  await page.screenshot({path:path.join(qa,'参数优化报告.png'),fullPage:true});
  if (errors.length) throw Error(errors.join('\n'));
  const result={page_errors:errors,defaults:true,input_validation:true,fractional_threshold_submission:true,mobile_no_overflow:true,report_rows:before,filtered_rows:after,artifact_download:true,api_mode:'actual read-only route fixtures; browser POST intercepted, actual subprocess tested separately'};
  fs.writeFileSync(path.join(qa,'browser-qa.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
