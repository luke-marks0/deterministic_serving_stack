Speaker B: Yeah. Yeah. So I think today it would be good if the first thing you do is write this paragraph on the networking stuff and take a look into that. And then once you've sent me that, get started on the memory wipes. And so it would probably be good for us now to discuss how the memory wipes should be implemented.

Speaker A: Yeah, let's do it.

Speaker B: Okay, so, um, like, do you have a sense for like, like roughly how proof of secure erasure works? Like, like an intuition for how it works?

Speaker A: Not really.

Speaker B: So basically the verifier knows like, okay, the prover has like a terabyte of storage or something. So I'm going to generate like 1 terabyte of noise and then I'm going to tell the prover like, overwrite like all of your storage with this noise. And then the prover is going to say, okay, I've overwritten it. And then the verifier is going to say, okay, well, if you're actually storing it in memory, then tell me what is like byte 7 million to byte 7 million and 5000 or whatever. And then the idea is like the prover can only respond to these challenges like in a timely manner if they were genuinely storing that object. Does that make sense?

Speaker A: I'm a bit confused because I thought that the prover was supposed to erase the data, not— yeah.

Speaker B: So the prover is supposed to erase what they previously had in memory, and the way that they do that is by overriding all of their memory with this noise.

Speaker A: I see, okay.

Speaker B: So you can imagine if I just tell you 100 numbers right now and all of your memory is just like an issue that can store 100 numbers and then I ask you, "What is the 50th number?" Unless you actually filled your paper with the numbers I told you, you're not going to be able to tell me what the 50th number is.

Speaker A: Yeah, got it.

Speaker B: But this is obviously complicated by the fact that the prover isn't just like some isolated computer with no other storage, right? Like it's going to be— it's going to have like an internet connection and be connected to other storage devices and stuff. So what we do is we leverage the fact that the prover and verifier have a higher bandwidth communication channel than the prover would with some other storage device somewhere else. Like you can imagine the verifier and prover can communicate at like 800 gigabits a second or something, but the prover and some other— and the internet can only communicate at like 10 gigabits a second. And so what this lets the prover— what this lets the verifier do is tell the prover, you have to respond in less than 0.1 seconds, because in less than 0.1 seconds, the prover wouldn't be able to pull all of the noise that the verifier told the prover to store from some other storage device. It would have to actually have stored those bits on its like, on its like, you know, HBM or like SSD or whatever, like on the actual, like its actual hardware. Does that make sense?

Speaker A: I'm sorry, could you repeat it one more time?

Speaker B: Yeah. So like what we're worried about is that like when the verifier says store this, like all of this noise.

Speaker A: Yeah.

Speaker B: The prover doesn't actually store that noise and they just tell like some other device to store that noise. And we're going to get around this attack because then they could just read the noise from that other device and respond to the verifier's challenges. So when the verifier is like, what is the 50th bit? The prover would just ask its other storage device, what is the 50th bit? And then it would send that to the verifier and then it would never actually have to store all the noise.

Speaker A: I understand that part.

Speaker B: Okay. Yeah. So what we're going to assume is that communication channel between the prover and the verifier is way higher bandwidth than the communication channel between the prover and any of its other storage devices. So you can imagine maybe the prover has some AWS instance with a bunch of storage on it, and it's going to try and put all of the noise that the verifier told it to store on the AWS instance instead of storing it locally. But then the verifier is going to say, actually, you have to tell me what the 50th random bit was. In under 0.1 seconds. And then because the prover and verifier communication channel is higher bandwidth, the prover is going to have time to send it from their local storage devices, but it's not going to have time to pull it from AWS.

Speaker A: I see. I see.

Speaker B: Okay.

Speaker A: So, okay. Is that— Is that realistic?

Speaker B: Yeah, so like, we can just like literally derive a really basic equation that shows like how, like, that shows that this is possible. Like if it's the case that like the prover and verifier's communication channel is like 100 times faster than the like than the communication channel between the prover and like its AWS instance, then like there's some equation in a paper that I'll send you that's just like, here is like how much data you have to ask the prover for such that you can be like 99.9% confident that like they're actually telling you that they were actually storing the noise locally. And like this is definitely realistic because the, like the storage fabric in data centers is like some of the highest bandwidth fiber in the world. So it's going to be way faster than any other type of connection.

Speaker A: Okay, makes sense so far.

Speaker B: Yeah, so how this is going to work, I think, in terms of actually implementing it, the main sort of challenge here is the prover has a lot of storage, right? Like the prover, In our actual demo, the Lambda Labs instance has maybe 128 gigs of host DRAM and a terabyte of NVMe SSD storage. And if it's an H100, it's going to have 80 gigs of HBM on the GPU. So that means that the verifier would have to store 1.2 or 3 terabytes of noise. Because it needs to know what that noise is when it streams it to the verifier. So I think how we implement this is the verifier is going to know a secret seed and a secret pseudorandom function that's going to generate all of the noise, so the prover doesn't know the seed and doesn't know the pseudorandom function. And then the verifier is going to be like, generating the output of the pseudorandom function live and streaming it to the prover. So this means that the verifier isn't going to have to actually store the random noise while streaming it to the prover. So you can imagine this is just some— the verifier knows some hash function that the prover doesn't know, and they're just going to generate some input for this hash function and then stream the output of this hash to the prover.

Speaker A: Okay, how is all the memory going to be— I'm just thinking about how the memory is all going to be overwritten because we still need memory for like the OS and stuff like that.

Speaker B: Yeah, so like We can't overwrite all of the memory, we can only overwrite a large fraction of the memory. So I've done some benchmarking of what this can be.

Speaker A: Cool.

Speaker B: So for GPU HBM, you have to leave about 128 megabytes on an H100 because a lot of the drivers and stuff live in HBM. And if you wipe anything more than that, the GPU just crashes, which we don't want. How did you do that? I have like a repo that I can send you that implements this. Awesome. And then the— and then for the prover, sorry, for the, for the host DRAM, you can wipe like everything except for like 512 megabytes or something. But we might want to leave like a bit more storage just so it's convenient. Like, I don't know, we just don't want to have the prover like having more than 2 gigabytes of storage in total. And then you can wipe literally almost all of an NVMe drive. There's a tiny amount that you can't wipe. Okay, got it. And yeah, so the main issue here though is the verifier at some point does actually have to store all of the noise because otherwise it's not going to be able to challenge 'cause you know when it's gonna say to the prover, like, send me the 1 billionth random bit, in order for the verifier to know what the 1 billionth random bit is, it actually has to have that stored somewhere. So the verifier is probably just gonna have to have a huge NVMe drive or something. And so how I would suggest implementing this is in the current demo where the prover and verifier are just separate instances, it would just take way too long to stream a terabyte size object over the internet, right? Because we want the wipes to take place really fast. We want them to be under a minute and streaming a terabyte object is probably going to take more than a minute.

Speaker A: Definitely.

Speaker B: And so instead what we're going to do is on our prover instance, we are going to set up a little like process that has access to an extra NVMe drive. So there'll be two, there'll be two NVMe drives on the prover, like on the prover instance. And one of them is going to be like a huge drive that's like bigger than the, like prover's HBM, their other, their main SSD drive and their host DRAM combined, right? So this NVMe drive has to be like 1.5 terabytes or something.

Speaker A: Okay.

Speaker B: And then this, this, this little verifier process, like it owns this drive. So like the rest of the, like of the prover instance cannot access this verifier drive. And then the verifier process is going to generate this, all of this noise and store it on this drive. And then they're going to stream the noise from this drive onto the prover's DRAM, HBM, and NVMe.

Speaker A: Okay. Okay.

Speaker B: Yeah. And then so I think that's not going to be too hard to implement, right? Because it's literally just like get the output of a hash function, stream it into DRAM, HBM, and NVMe. I think what is going to be harder is figuring out how to get the prover to be able to resume their inference workload after a wipe. But I think we can just worry about that tomorrow or the day after tomorrow, just whenever you get the initial wipe working. So what I would do right now is just have a like make your Lambda Labs instances have an additional NVMe drive, figure out how to get the verifier process set up and owning this drive and generating the noise, and then figure out how to get that to stream to the DRAM, HBM, and NVMe. And like that right now, that will just like crash the workload, but like that's fine. We can just worry about like resuming the workload once we have all of the basics set up.

Speaker A: Yeah, got it. Um, okay. Um, I think— is that, is that all?

Speaker B: Yeah. So like, what would be, what would be great, um, for tomorrow, um, like, is if you have the, um, like, analysis of the networking stuff done, and if you have the, like, memory wipes basic done. Basics done. So this just includes like the verify process, owning the NVMe drive, and streaming all the noise to the, to the approver's storage devices.

Speaker A: Okay, is it okay if you give me like a day more for the basic memory wipes? I know it's like, um, it shouldn't be that much effort, but I want to take the time to, um, really understand the, uh, I really understand the correctness of the process.

Speaker B: Yeah, yeah, no, I think it would be fine if it took an extra day to do the memory wipe stuff.

Speaker A: Okay.

Speaker B: Just like let me know where your progress is at with that tomorrow.

Speaker A: Yeah, and then I'll do the networking stuff today for sure.

Speaker B: Okay, all right, sick. Um, okay, I think that should be everything then. Oh yeah, and I will, like, no obligation to you to read this, but I will just send you the paper draft because it might just have like some information that like is useful to you. Like it basically has a write-up of like how the memory wipes should be implemented and stuff like that. So it might just be helpful to have access to that.

Speaker A: Yep, it sounds great. Okay, can you also send me— you may have sent it to me before, but can you send me again the paper about the proof of erasure?

Speaker B: Yeah, yeah, I'll send that to you as well.

Speaker A: Proof of erasure, then the paper draft. Okay, and then I think that's about it, and then we'll talk tomorrow.

Speaker B: Yeah.

Speaker A: Okay, I mean, I probably have more— I'll definitely have more questions over the course of today.

Speaker B: Um, okay.

Speaker A: Yeah, but I think it's fine. Okay, um, then I'll talk to you later.

Speaker B: Okay, sounds good.

Speaker A: Awesome. See ya. See ya.