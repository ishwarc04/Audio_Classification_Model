## ∙ Team Name
EcoByte Labs

---
## ∙ AWS Builder Center Aliases (all members)
- aadi2006  
---

## ∙ Track
Sustainability & Green Innovation

---

## ∙ Problem we’re solving

Organizations must report **Scope-3 emissions** under frameworks like ESG reporting and **SEBI BRSR**.  
A major portion comes from **employee commuting and work-from-home electricity usage**, yet companies rely on **self-reported surveys and rough estimates** that are inaccurate and difficult to audit.

At the same time, most carbon-tracking apps require **manual activity logging**, leading to **user fatigue and low adoption**.

There is currently **no scalable system that can automatically generate reliable household-level carbon data.**

---

## ∙ Our proposed solution

**Scope3Sense** is a cyber-physical sustainability platform connecting **household energy data with corporate carbon accounting**.

Our system combines:

- **IoT energy sensing**
- **AI activity logging**
- **AWS cloud infrastructure**

to produce **verifiable carbon insights for individuals and organizations**.

Instead of manual inputs, we capture:

- **real electricity telemetry from an IoT device**
- **daily activities via a chatbot (WhatsApp / Telegram)**

Data is processed using AWS services to generate:

- real-time carbon insights for individuals  
- aggregated Scope-3 emissions data for organizations

### Hackathon MVP

We will deploy a working prototype consisting of:

- **ESP32 energy node streaming data to AWS IoT Core**
- **chatbot logging commuting and activity emissions**
- **basic sustainability dashboard**

---

## System Architecture

*(Insert architecture diagram image here)*

**Core AWS services**

- AWS IoT Core  
- Amazon API Gateway  
- AWS Lambda  
- Amazon EC2  
- Amazon RDS  
- Amazon Timestream  
- Amazon SageMaker  
- AWS Amplify

---

## ∙ Key features we plan to build

**Hardware-Verified Energy Tracking**

- ESP32 node measures household electricity usage  
- data streamed through **AWS IoT Core**

**Single-Sensor Appliance Detection**

- **Non-Intrusive Load Monitoring (NILM)** identifies appliance usage  
- works with **one central sensor**

**AI Carbon Logging**

- chatbot records activities via **WhatsApp / Telegram**  
- text, voice, or bill photos converted into carbon logs

**Energy Forecasting**

- ML model predicts **24-hour household energy demand**

**Scope-3 Dashboard**

- aggregated sustainability insights for organizations

---

## ∙ What makes it different from existing tools

**Hardware-verified carbon data**  
Most tools rely on estimates. We combine **IoT sensing + AI** to capture real energy usage.

**Zero-friction experience**  
Carbon tracking happens through **chat platforms people already use**.

**Single-sensor scalability**  
NILM enables appliance insights from **one device**, lowering deployment cost.

**Bridging individuals and ESG reporting**  
Our platform links **household behavior with corporate Scope-3 reporting.**