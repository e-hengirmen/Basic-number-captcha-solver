import time
from os.path import exists
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait

from selenium.webdriver import Chrome
from webdriver_manager.chrome import ChromeDriverManager

import urllib.request

#########################################################
# change myDEPT to your department
# delete departments that you dont want to take courses from the class_codes list
Username    = "e123456"     #fill your metu username
Password    = "#########"    #fill your password
#########################################################

start = time.time()

# opening course window
url  = "https://student.metu.edu.tr/"
url2 = "https://images.google.com"
driver =  webdriver.Chrome(ChromeDriverManager().install())
driver.maximize_window()
driver.get(url)
driver.find_element(By.LINK_TEXT,"View Course Capacity (158)").click()
WebDriverWait(driver,10).until(EC.presence_of_element_located((By.ID,"textUsername"))).send_keys(Username)
WebDriverWait(driver,10).until(EC.presence_of_element_located((By.ID,"textPassword"))).send_keys(Password)
WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH,'//*[@id="signinForm"]/fieldset/div[3]/div/button[1]'))).click()



driver.switch_to.new_window('tab')
driver.switch_to.window(driver.window_handles[0])




iframe = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))  # its in the first frame
driver.switch_to.frame(iframe)

course_code="5710140"
fail_count=0

number_of_samples=10000
checkpoint=number_of_samples//100
if(checkpoint==0):
    checkpoint=1
for i in range(1,number_of_samples+1):
    captcha_fails = True

    while captcha_fails:
        # change to frame enter course code and get image url

        captcha_url = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
            (By.XPATH, '//*[@id="SignInFormDiv"]/form/fieldset/div[2]/div[1]/img'))).get_attribute("src")

        # find captcha result from basic google image search
        driver.switch_to.window(driver.window_handles[1])
        driver.get(url2)
        driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[3]/div[4]").click()
        WebDriverWait(driver, 10).until(EC.presence_of_element_located(
            (By.XPATH, '//*[@id="ow6"]/div[3]/c-wiz/div[2]/div/div[3]/div[2]/c-wiz/div[2]/input'))).send_keys(
            captcha_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located(
            (By.XPATH, '//*[@id="ow6"]/div[3]/c-wiz/div[2]/div/div[3]/div[2]/c-wiz/div[2]/div'))).click()
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="ucj-5"]/span[1]'))).click()
        try:
            captcha_result = WebDriverWait(driver, 4).until(EC.presence_of_element_located((By.XPATH,'//*[@id="yDmH0d"]/div[3]/c-wiz/div/c-wiz/c-wiz/div/div[2]/div/div/div/div[1]/div/div[3]/div/div/div'))).text

        except:
            captcha_result = "0"

        captcha_result = ''.join(filter(str.isdigit,captcha_result))[0:6]

        driver.switch_to.window(driver.window_handles[0])
        if len(captcha_result)!=6 :
            captcha_result="0"

        driver.switch_to.frame(iframe)
        input_course = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "text_course_code")))
        input_captcha = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "text_img_number")))
        input_course.clear()
        input_course.send_keys(course_code)
        input_captcha.clear()
        input_captcha.send_keys(captcha_result)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="SignInFormDiv"]/form/fieldset/div[4]/div[2]/input'))).click()

        if WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,'/html/body/div/div/div[1]/div[1]'))).text=="×\nInvalid Image Verification.":
            fail_count+=1
            continue
        captcha_fails=False

        sections=WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
            (By.XPATH, '/html/body/div/div/div[1]/div/div[2]/div/div[3]/div/form/table/tbody/tr')))[0:]
        urllib.request.urlretrieve(captcha_url, './captchas/'+str(i)+'-'+str(captcha_result)+'.jpg')

        if(i%checkpoint==0):
            print(str(i/number_of_samples*100)+"%")
        ### check failure if does not fail

print(fail_count)
driver.quit()
quit()

#################################################################
