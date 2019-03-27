package com.nigam.ancache.core;

import com.nigam.ancache.model.CacheElement;

public class ANCacheRunner {

	public static void main(String ...strings) throws InterruptedException {

		ANCacheManager<Integer, String> cache = new ANCacheManager<>(30, 5);
		
		CacheElement<Integer, String> i = cache.add(1, "Abhinav");
		System.out.println(i.getK());
		System.out.println(i.getV());
		
		Thread.sleep(25);
		
		CacheElement<Integer, String> x = cache.get(1);
		System.out.println(x.getK());
		System.out.println(x.getV());
		System.out.println(i.equals(x));
		
		
		x = cache.get(100);
		System.out.println(x.getK());
		System.out.println(x.getV());
		System.out.println(i.equals(x));
	}

}
